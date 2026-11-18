# DINOv3 six-source fine-tuning and fifteen pair averages.
# Stop when consecutive validation accuracy improves by less than 0.5 points.
# Existing completed jobs remain complete. Existing partial jobs resume.
# The legacy threshold setting remains in config.json for Colab compatibility.
import csv
import hashlib
import itertools
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader

import dino_common as base

DOMAINS = tuple(base.DOMAINS)
PAIRS = tuple(itertools.combinations(DOMAINS, 2))
MIN_IMPROVEMENT = 0.005
POLICY = 'validation_improvement_below_0.5_percentage_points'
MODEL_REPO = 'timm/vit_small_patch16_dinov3.lvd1689m'
MODEL_REVISION = '3bf4720a82ec2066db88137180ff1f83a675cef0'
MODEL_SHA256 = '2a1ec16ae28ffa07bc0ead0241ee7df9fc26451fe6f9f839b7b3afa0a906b040'


# Calculate a file checksum for provenance and checkpoint identity.
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Write a CSV summary through a temporary file.
def table(path, rows, columns):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


# Build experiment settings compatible with the existing Colab run.
def config(data, out, **overrides):
    values = dict(data=Path(data), out=Path(out), seed=42, val_fraction=.10,
                  max_source_epochs=50, batch_size=32, workers=2,
                  backbone_lr=1e-5, head_lr=1e-3, weight_decay=.05,
                  threshold=.90, device='cuda')
    values.update(overrides)
    if values['max_source_epochs'] < 1 or values['batch_size'] < 1:
        raise ValueError('Epoch limit and batch size must be positive')
    return SimpleNamespace(**values)


# Fetch the pinned timm release and verify its safetensors checksum.
def download_pretrained(directory):
    from huggingface_hub import hf_hub_download
    directory.mkdir(parents=True, exist_ok=True)
    for name in ('model.safetensors', 'config.json', 'LICENSE.md', 'README.md'):
        downloaded = Path(hf_hub_download(MODEL_REPO, name, revision=MODEL_REVISION,
                                          local_dir=directory))
        if name == 'model.safetensors' and digest(downloaded) != MODEL_SHA256:
            raise ValueError('Pretrained model checksum mismatch')
    return directory / 'model.safetensors'


# Verify settings and splits, then create the common initialization once.
def prepare(args, pretrained):
    from safetensors.torch import load_file
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()}
    settings.update(model_repo=MODEL_REPO, model_revision=MODEL_REVISION,
                    pretrained_sha256=digest(pretrained), architecture='DINOv3 ViT-S/16',
                    image_size=256, pooling='avg', validation_every_epochs=1)
    path = args.out / 'config.json'
    if path.exists() and json.loads(path.read_text()) != settings:
        raise ValueError('Configuration changed. Choose a new results directory.')
    base.write_json(path, settings)
    splits = {}
    split_hashes = {}
    for domain in DOMAINS:
        train_rows = base.read_split(args.data, domain, 'train')
        test_rows = base.read_split(args.data, domain, 'test')
        if {r[0] for r in train_rows} & {r[0] for r in test_rows}:
            raise ValueError(f'Train/test overlap in {domain}')
        train, val = base.partition(train_rows, args.val_fraction, args.seed)
        splits[domain] = dict(train=train, val=val, test=test_rows)
        for split in ('train', 'test'):
            name = f'{domain}_{split}.txt'
            split_hashes[name] = digest(args.data / 'splits' / name)
    manifest = dict(partitions=splits, split_sha256=split_hashes, seed=args.seed)
    path = args.out / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != json.loads(json.dumps(manifest)):
        raise ValueError('Saved data membership changed')
    base.write_json(path, manifest)
    initial = args.out / 'initial.pt'
    if not initial.exists():
        base.seed_all(args.seed)
        model = base.make_model()
        model[0].load_state_dict(load_file(str(pretrained)), strict=True)
        base.save_checkpoint(initial, base.cpu_state(model), {
            'pretrained_sha256': digest(pretrained), 'seed': args.seed})
    counts = [{'domain': d, **{k: len(v) for k, v in rows.items()}}
              for d, rows in splits.items()]
    table(args.out / 'split_counts.csv', counts, ['domain', 'train', 'val', 'test'])
    return splits


# Resume one domain job and stop on insufficient future validation improvement.
def fit(args, rows, initialization, directory, max_epochs, identity):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'result.json'
    if path.exists():
        return json.loads(path.read_text())
    job = dict(identity=identity, initialization_sha256=digest(initialization),
               max_epochs=max_epochs, run_config_sha256=digest(args.out / 'config.json'),
               manifest_sha256=digest(args.out / 'manifest.json'))
    job_path = directory / 'job.json'
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError(f'Changed job configuration for {identity}')
    base.write_json(job_path, job)
    base.seed_all(args.seed)
    model = base.make_model().to(args.device)
    model.load_state_dict(base.load_checkpoint(initialization)['state'], strict=True)
    optimizer = torch.optim.AdamW([
        {'params': model[0].parameters(), 'lr': args.backbone_lr},
        {'params': model[1].parameters(), 'lr': args.head_lr}], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_source_epochs)
    use_cuda = args.device.startswith('cuda')
    scaler = torch.amp.GradScaler('cuda', init_scale=1024., enabled=use_cuda)
    validation = DataLoader(base.Images(args.data, rows['val']), batch_size=args.batch_size,
        num_workers=args.workers, persistent_workers=args.workers > 0, pin_memory=use_cuda)
    history = []
    steps, train_seconds, previous_wall = 0, 0., 0.
    best = -1.
    first90 = None
    next_epoch = 1
    resume = directory / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        history = saved['history']
        steps, train_seconds = saved['steps'], saved['train_seconds']
        best = saved['best']
        first90 = saved.get('first90') if saved.get('stopping_criterion') == POLICY else None
        previous_wall = saved['wall_seconds']
        next_epoch = saved['epoch'] + 1
        torch.set_rng_state(saved['torch_rng'])
        random.setstate(saved['python_rng'])
        if use_cuda:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    start = time.perf_counter()

    # Validate, update the stopping decision, and checkpoint the completed epoch.
    def observe(epoch, train_accuracy=None):
        nonlocal best, first90
        metrics = base.evaluate(model, validation, args.device)
        wall_seconds = previous_wall + time.perf_counter() - start
        row = dict(epoch=epoch, optimizer_steps=steps, train_seconds=train_seconds,
                   wall_seconds=wall_seconds, validation_accuracy=metrics['accuracy'],
                   validation_loss=metrics['loss'], augmented_train_accuracy=train_accuracy)
        history.append(row)
        if metrics['accuracy'] > best:
            best = metrics['accuracy']
            base.save_checkpoint(directory / 'best.pt', base.cpu_state(model), {
                'identity': identity, 'epoch': epoch, 'validation_accuracy': best,
                'initial_sha256': job['initialization_sha256']})
        improvement = None
        if epoch > 0 and len(history) >= 2:
            improvement = metrics['accuracy'] - history[-2]['validation_accuracy']
            if improvement < MIN_IMPROVEMENT - 1e-12 and first90 is None:
                first90 = epoch
        row['validation_improvement'] = improvement
        temporary = resume.with_suffix('.tmp')
        torch.save(dict(model=base.cpu_state(model), optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                        history=history, stopping_criterion=POLICY, steps=steps, train_seconds=train_seconds,
                        best=best, first90=first90, epoch=epoch, wall_seconds=wall_seconds,
                        torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                        cuda_rng=torch.cuda.get_rng_state_all() if use_cuda else []), temporary)
        temporary.replace(resume)
        base.write_json(directory / 'history.json', history)
        base.write_json(directory / 'progress.json', row)
        print(identity, row, flush=True)

    if not history:
        observe(0)
    for epoch in range(next_epoch, max_epochs + 1):
        if first90 is not None:
            break
        batches = DataLoader(base.Images(args.data, rows['train'], True),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
            pin_memory=use_cuda, generator=torch.Generator().manual_seed(args.seed + epoch))
        model.train()
        hits, count, updates = 0, 0, 0
        if use_cuda:
            torch.cuda.synchronize()
        epoch_started = time.perf_counter()
        for batch_index, (x, y) in enumerate(batches, 1):
            optimizer.zero_grad(set_to_none=True)
            y = y.to(args.device)
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_cuda):
                output = model(x.to(args.device))
                loss = nn.functional.cross_entropy(output, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {identity}')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            successful = int(scaler.get_scale() >= scale)
            steps += successful
            updates += successful
            hits += (output.argmax(1) == y).sum().item()
            count += len(y)
            if batch_index % 100 == 0:
                print(identity, 'epoch', epoch, 'batch', batch_index, '/', len(batches),
                      'loss', round(loss.item(), 4), flush=True)
        if use_cuda:
            torch.cuda.synchronize()
        train_seconds += time.perf_counter() - epoch_started
        if updates:
            scheduler.step()
        observe(epoch, hits / count)
    model.load_state_dict(base.load_checkpoint(directory / 'best.pt')['state'])
    test = base.evaluate(model, base.loader(args, rows['test']), args.device)
    result = dict(identity=identity, status='converged' if first90 is not None else 'budget_exhausted',
                  epochs_to_convergence=first90, epochs_to_90=first90,
                  stopping_criterion=POLICY, min_improvement=MIN_IMPROVEMENT, epochs_completed=history[-1]['epoch'],
                  max_epochs=max_epochs, best_validation_accuracy=best,
                  test_accuracy=test['accuracy'], optimizer_steps=steps,
                  train_seconds=train_seconds, wall_seconds=previous_wall + time.perf_counter() - start)
    base.write_json(path, result)
    # Completed jobs retain their best model and histories. Only the optimizer
    # resume file is removed to keep all 66 runs within practical Drive storage.
    resume.unlink(missing_ok=True)
    print('Finished', result, flush=True)
    return result


# Fine-tune six standalone models and refresh the convergence summary.
def source_runs(args, splits):
    for domain in DOMAINS:
        try:
            fit(args, splits[domain], args.out / 'initial.pt', args.out / 'sources' / domain,
                args.max_source_epochs, domain)
        finally:
            report(args)


# Average a source pair with equal weights and verify source identities.
def averaged_checkpoint(args, pair):
    directory = args.out / 'pair_initializations'
    directory.mkdir(exist_ok=True)
    destination = directory / ('__'.join(pair) + '.pt')
    paths = [args.out / 'sources' / d / 'best.pt' for d in pair]
    metadata = {'source_domains': list(pair), 'source_sha256': [digest(p) for p in paths],
                'method': 'uniform_parameter_average', 'weights': [.5, .5]}
    if destination.exists():
        if base.load_checkpoint(destination)['metadata'] != metadata:
            raise ValueError('Source weights changed for an existing averaged checkpoint')
        return destination
    states = [base.load_checkpoint(p)['state'] for p in paths]
    if states[0].keys() != states[1].keys():
        raise ValueError('Incompatible source states')
    averaged = {}
    for name, first in states[0].items():
        second = states[1][name]
        if first.shape != second.shape or first.dtype != second.dtype:
            raise ValueError(f'Incompatible tensor {name}')
        if first.is_floating_point():
            averaged[name] = ((first.float() + second.float()) / 2).to(first.dtype)
        elif torch.equal(first, second):
            averaged[name] = first.clone()
        else:
            raise ValueError(f'Different nonfloating buffer {name}')
    base.save_checkpoint(destination, averaged, metadata)
    return destination


# Adapt each source-pair average to its four remaining target domains.
def pair_runs(args, splits, selected_pairs=None):
    for pair in PAIRS if selected_pairs is None else selected_pairs:
        if tuple(pair) not in PAIRS:
            raise ValueError(f'Pair must follow the displayed domain order: {pair}')
        checkpoint = averaged_checkpoint(args, pair)
        for target in DOMAINS:
            if target in pair:
                continue
            reference = args.out / 'sources' / target / 'result.json'
            if not reference.exists():
                raise ValueError(f'Run the standalone {target} model first')
            reference_result = json.loads(reference.read_text())
            reference = reference_result.get('epochs_to_convergence', reference_result.get('epochs_to_90'))
            if reference is None:
                print(pair, target, 'reference_unavailable: standalone did not meet its stopping criterion', flush=True)
                continue
            try:
                fit(args, splits[target], checkpoint, args.out / 'pairs' / '__'.join(pair) / target,
                    reference, '__'.join(pair) + ' -> ' + target)
            finally:
                report(args)
    report(args)


# Summarize source and pair convergence with legacy result compatibility.
def report(args):
    sources = []
    for domain in DOMAINS:
        path = args.out / 'sources' / domain / 'result.json'
        value = json.loads(path.read_text()) if path.exists() else {'status': 'pending'}
        value['epochs_to_convergence'] = value.get('epochs_to_convergence', value.get('epochs_to_90'))
        value['stopping_criterion'] = value.get('stopping_criterion', 'legacy_90_percent' if path.exists() else None)
        sources.append(dict(domain=domain, **value))
    table(args.out / 'source_convergence.csv', sources, ['domain', 'status', 'stopping_criterion', 'epochs_to_convergence',
          'epochs_completed', 'best_validation_accuracy', 'test_accuracy', 'train_seconds'])
    rows = []
    references = {row['domain']: row.get('epochs_to_convergence') for row in sources}
    completed = {row['domain']: row['status'] != 'pending' for row in sources}
    for pair in PAIRS:
        for target in DOMAINS:
            if target in pair:
                continue
            reference = references[target]
            path = args.out / 'pairs' / '__'.join(pair) / target / 'result.json'
            result = json.loads(path.read_text()) if path.exists() else {}
            status = result.get('status', 'reference_unavailable' if completed[target] and reference is None else 'pending')
            epoch = result.get('epochs_to_convergence', result.get('epochs_to_90'))
            rows.append(dict(source_a=pair[0], source_b=pair[1], target=target,
                status=status, stopping_criterion=result.get('stopping_criterion'), standalone_epochs_to_convergence=reference, averaged_epochs_to_convergence=epoch,
                epochs_saved=reference-epoch if reference is not None and epoch is not None else None,
                best_validation_accuracy=result.get('best_validation_accuracy'),
                test_accuracy=result.get('test_accuracy'), train_seconds=result.get('train_seconds')))
    table(args.out / 'pair_convergence.csv', rows, ['source_a', 'source_b', 'target', 'status', 'stopping_criterion',
          'standalone_epochs_to_convergence', 'averaged_epochs_to_convergence', 'epochs_saved',
          'best_validation_accuracy', 'test_accuracy', 'train_seconds'])
