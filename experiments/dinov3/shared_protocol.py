"""Shared DomainNet sampling and target adaptation; no legacy run is modified."""
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

VERSION = 'domainnet-shared-v1'
DOMAINS = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')
DATASET = 'wltjr1007/DomainNet'
REVISION = 'ee20570ae7a29c51571e55a9a17983f7625295d6'
SETTINGS = dict(batch_size=64, image_size=224, backbone_lr=3e-5, head_lr=1e-3,
                weight_decay=.05, warmup_steps=100, eval_every=100,
                max_epochs=10, patience_epochs=2, min_gain_points=.5,
                relative_threshold=.95, precision='float32', drop_last=True)


# Hash canonical JSON rather than platform-dependent whitespace or path separators.
def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


# Stream large checkpoint files while computing their identity.
def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


# Replace a complete JSON document atomically, retaining the previous file on failure.
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


# Validate official split membership and preserve the original parquet row order.
def read_rows(root, domain, split):
    rows = []
    for line in (Path(root) / 'splits' / f'{domain}_{split}.txt').read_text().splitlines():
        name, label = line.rsplit(maxsplit=1)
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.parts[0] != domain:
            raise ValueError(f'Invalid image path: {name}')
        label = int(label)
        if not 0 <= label < 345:
            raise ValueError(f'Invalid class: {label}')
        rows.append([path.as_posix(), label])
    if not rows or len({r[0] for r in rows}) != len(rows):
        raise ValueError(f'Empty or duplicate split: {domain}/{split}')
    return rows


# Match fine-tuning's seeded class sampler; report classes with fewer than 50 images.
def sample_classes(rows, count=50, seed=0):
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    rng = random.Random(seed)
    selected = [row for label in sorted(groups) for row in
                (groups[label] if len(groups[label]) <= count else rng.sample(groups[label], count))]
    shortfalls = {str(label): len(groups.get(label, [])) for label in range(345)
                  if len(groups.get(label, [])) < count}
    return selected, shortfalls


# Persist the actual filenames so every branch can use the identical images.
def make_manifest(root, path, seed=0):
    root, path = Path(root), Path(path)
    provenance = json.loads((root / 'provenance.json').read_text())
    if provenance.get('repo') != DATASET or provenance.get('revision') != REVISION:
        raise ValueError('Dataset provenance does not match the pinned DomainNet release')
    document = dict(version=VERSION, dataset=DATASET, revision=REVISION, seed=seed,
                    per_class=50, test_images=2000, domains={})
    for domain in DOMAINS:
        train, test = read_rows(root, domain, 'train'), read_rows(root, domain, 'test')
        if {r[0] for r in train} & {r[0] for r in test}:
            raise ValueError(f'Train/test overlap: {domain}')
        chosen, shortfalls = sample_classes(train, seed=seed)
        document['domains'][domain] = dict(
            train=chosen, test=random.Random(seed).sample(test, 2000),
            train_class_counts=dict(sorted(Counter(str(r[1]) for r in chosen).items())),
            classes_below_50=shortfalls,
            official_train_hash=fingerprint(train), official_test_hash=fingerprint(test))
    document['sha256'] = fingerprint(document)
    if path.exists() and json.loads(path.read_text()) != document:
        raise ValueError('Manifest changed; use a new manifest path and run directory')
    if not path.exists():
        write_json(path, document)
    return document


# Check a shared manifest against the local pinned splits before training.
def load_manifest(root, path):
    document = json.loads(Path(path).read_text())
    if document.get('sha256') != fingerprint({k: v for k, v in document.items() if k != 'sha256'}):
        raise ValueError('Manifest checksum mismatch')
    expected = make_manifest(root, path, document['seed'])
    if document != expected:
        raise ValueError('Manifest does not match this dataset')
    return document


# Use the reference branch's two-epoch, 0.5-percentage-point plateau window.
def plateau(curve, steps_per_epoch, settings):
    step = curve[-1][0]
    boundary = step - math.ceil(settings['patience_epochs'] * steps_per_epoch)
    old = [accuracy for s, accuracy in curve if s <= boundary]
    recent = [accuracy for s, accuracy in curve if s > boundary]
    return bool(old and recent) and max(recent) < max(old) + settings['min_gain_points']


# Measure the first observed crossing of 95% of the best observed accuracy.
def summarize(curve, stopped_by, settings):
    best = max(accuracy for _, accuracy in curve)
    threshold = settings['relative_threshold'] * best
    reached = next(step for step, accuracy in curve if accuracy >= threshold)
    return dict(best_accuracy=best, threshold_accuracy=threshold, converged_at=reached,
                stopped_by=stopped_by, plateau_confirmed=stopped_by == 'plateau',
                budget_censored=stopped_by == 'budget', curve=curve)


# Load either this experiment's Sequential model or another branch's timm model.
def load_weights(model, path):
    import torch
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved
    for key in ('state', 'model', 'state_dict'):
        if isinstance(saved, dict) and isinstance(saved.get(key), dict):
            state = saved[key]
            break
    expected = model.state_dict()
    if set(state) != set(expected):
        converted = {}
        for key, value in state.items():
            key = key.removeprefix('module.')
            converted[('1.' + key[5:]) if key.startswith('head.') else '0.' + key] = value
        state = converted
    model.load_state_dict(state, strict=True)


# Average every matching floating tensor, including the classifier; reject unsafe buffers.
def average_states(first, second):
    import torch
    if first.keys() != second.keys():
        raise ValueError('Source architectures differ')
    averaged = {}
    for key, left in first.items():
        right = second[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f'Incompatible tensor: {key}')
        if left.is_floating_point():
            averaged[key] = ((left.float() + right.float()) / 2).to(left.dtype)
        elif torch.equal(left, right):
            averaged[key] = left.clone()
        else:
            raise ValueError(f'Non-floating buffers differ: {key}')
    return averaged


# Keep the dataset class importable so Windows and spawn-based workers can pickle it.
class ManifestImages:
    # Build the reference transforms and retain the augmentation stream identity.
    def __init__(self, root, rows, training, seed, epoch):
        from torchvision import transforms as T
        self.root, self.rows, self.seed, self.epoch = Path(root), rows, seed, epoch
        ops = ([T.RandomResizedCrop(224, (.5, 1), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip()] if training else
               [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])

    # Return the number of explicitly selected examples.
    def __len__(self):
        return len(self.rows)

    # Seed transforms by sample and epoch, independent of worker prefetch and restart.
    def __getitem__(self, index):
        import torch
        from PIL import Image
        name, label = self.rows[index]
        with Image.open(self.root / name) as image:
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self.seed + self.epoch * 1000003 + index)
                return self.transform(image.convert('RGB')), label


# Create the importable image dataset used by training and fixed-list evaluation.
def image_dataset(root, rows, training, seed, epoch=0):
    return ManifestImages(root, rows, training, seed, epoch)


# Adapt an arbitrary compatible initialization with a fresh optimizer and the shared recipe.
def train(model, root, rows, directory, identity, device='cuda', workers=2,
          seed=0, settings=None, source=False, interrupt_after=None):
    import time
    import torch
    from torch.utils.data import DataLoader

    settings = dict(SETTINGS if settings is None else settings)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    job = dict(version=VERSION, identity=identity, settings=settings, seed=seed,
               source=source, device=device, torch_version=str(torch.__version__),
               train_hash=fingerprint(rows['train']), test_hash=fingerprint(rows['test']))
    job_path = directory / 'job.json'
    if job_path.exists() and json.loads(job_path.read_text()) != job:
        raise ValueError(f'Changed protocol or initialization in {directory}')
    write_json(job_path, job)
    result_path = directory / 'result.json'
    if result_path.exists():
        return json.loads(result_path.read_text())
    if settings['precision'] not in ('float32', 'bfloat16', 'float16'):
        raise ValueError('Unknown precision')
    cuda = str(device).startswith('cuda')
    if settings['precision'] != 'float32' and not cuda:
        raise ValueError('Mixed precision requires CUDA in this protocol')
    if settings['precision'] == 'bfloat16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Select float16 on GPUs without bfloat16 support; use it for all methods')
    # Avoid hardware-dependent TF32 defaults and record explicit precision in every job.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
    model.to(device)
    head = list(model[1].parameters())
    optimizer = torch.optim.AdamW([
        {'params': model[0].parameters(), 'lr': settings['backbone_lr']},
        {'params': head, 'lr': settings['head_lr']}], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min((s + 1) / settings['warmup_steps'], 1.))
    scaler = torch.amp.GradScaler('cuda', enabled=cuda and settings['precision'] == 'float16')
    dtype = torch.bfloat16 if settings['precision'] == 'bfloat16' else torch.float16
    batch = settings['batch_size']
    per_epoch = len(rows['train']) // batch
    if per_epoch < 1:
        raise ValueError('Target sample is smaller than one full batch')
    cap = settings['max_epochs'] * per_epoch
    evaluation = DataLoader(image_dataset(root, rows['test'], False, seed),
                            batch_size=256, num_workers=workers, pin_memory=cuda)
    curve, step, elapsed = [], 0, 0.
    resume = directory / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['job_hash'] != fingerprint(job):
            raise ValueError('Resume checkpoint belongs to another protocol')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        curve, step, elapsed = saved['curve'], saved['step'], saved['elapsed']
        torch.set_rng_state(saved['rng'])
        if cuda:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    started = time.perf_counter()

    # Evaluate the fixed official-test monitoring list; accuracies are percentage points.
    @torch.inference_mode()
    def accuracy():
        model.eval()
        hits, total = 0, 0
        for x, y in evaluation:
            with torch.autocast('cuda', dtype=dtype, enabled=cuda and settings['precision'] != 'float32'):
                logits = model(x.to(device))
            hits += (logits.argmax(1).cpu() == y).sum().item()
            total += len(y)
        model.train()
        return 100 * hits / total

    # Save optimizer, scaler, RNG, batch cursor and metrics without deleting earlier checkpoints.
    def checkpoint():
        payload = dict(model={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                       optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                       scaler=scaler.state_dict(), curve=curve, step=step,
                       elapsed=elapsed + time.perf_counter() - started,
                       rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else [],
                       job_hash=fingerprint(job))
        temporary = resume.with_suffix('.pt.tmp')
        torch.save(payload, temporary)
        temporary.replace(resume)
        write_json(directory / 'history.json', curve)

    if not curve and not source:
        curve.append([0, accuracy()])
        checkpoint()
    stopped = 'budget'
    # A checkpoint may have been saved at the stopping evaluation before result.json was written.
    done = not source and len(curve) > 1 and plateau(curve, per_epoch, settings)
    if done:
        stopped = 'plateau'
    while step < cap and not done:
        epoch, offset = divmod(step, per_epoch)
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(rows['train']), generator=generator).tolist()
        order = order[offset * batch:per_epoch * batch]
        dataset = image_dataset(root, rows['train'], True, seed, epoch)
        batches = DataLoader(dataset, batch_size=batch, sampler=order, drop_last=True,
                             num_workers=workers, pin_memory=cuda,
                             generator=torch.Generator().manual_seed(seed + epoch))
        model.train()
        for x, y in batches:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype, enabled=cuda and settings['precision'] != 'float32'):
                loss = torch.nn.functional.cross_entropy(model(x.to(device)).float(), y.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss before step {step + 1}')
            scaler.scale(loss).backward()
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                raise FloatingPointError('AMP skipped an optimizer update; resume from last checkpoint')
            scheduler.step()
            step += 1
            if not source and (step % settings['eval_every'] == 0 or step == cap):
                curve.append([step, accuracy()])
                done = plateau(curve, per_epoch, settings)
                if done:
                    stopped = 'plateau'
                print(f'{directory.name}: step {step}/{cap}, accuracy {curve[-1][1]:.2f}%', flush=True)
                checkpoint()
            elif step % per_epoch == 0:
                checkpoint()
                print(f'{directory.name}: completed epoch {step // per_epoch}', flush=True)
            if interrupt_after is not None and step == interrupt_after:
                checkpoint()
                raise InterruptedError('Requested verification interruption')
            if done or step >= cap:
                break
    checkpoint()
    result = dict(job=job, steps=step, steps_per_epoch=per_epoch, n_train=len(rows['train']),
                  n_test=len(rows['test']), elapsed_seconds=elapsed + time.perf_counter() - started)
    if not source:
        result.update(summarize(curve, stopped, settings))
    else:
        result.update(stopped_by='fixed_source_budget', curve=[])
    # Final source weights are selected without looking at the official test monitoring set.
    temporary = directory / 'final.pt.tmp'
    torch.save({'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                'metadata': dict(job_hash=fingerprint(job), steps=step)}, temporary)
    temporary.replace(directory / 'final.pt')
    write_json(result_path, result)
    return result
