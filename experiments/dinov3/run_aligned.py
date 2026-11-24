"""Six 50-shot sources, 15 pair averages, 60 shared-protocol target adaptations."""
import argparse
import csv
import itertools
import json
from pathlib import Path

import torch

import dino_common as base
import dino_data
import dino_epochs as legacy
import shared_protocol as shared


# Refuse to reuse an old run or silently change a scientific setting on resume.
def configure(args, manifest):
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / 'config.json').exists() or (args.out / 'source_convergence.csv').exists():
        raise ValueError('This is a legacy run directory; choose a NEW --out path')
    config = dict(version=shared.VERSION, manifest_sha256=manifest['sha256'], seed=manifest['seed'],
                  source_epochs=args.source_epochs, settings=dict(shared.SETTINGS, precision=args.precision),
                  model_repo=legacy.MODEL_REPO, model_revision=legacy.MODEL_REVISION,
                  pretrained_sha256=legacy.MODEL_SHA256, pooling='avg', classes=345,
                  source_selection='final_fixed_epoch_weights_without_test_selection')
    path = args.out / 'aligned_config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Run configuration changed; choose a new output directory')
    shared.write_json(path, config)
    return config


# Initialize one common pretrained backbone and classifier before training the six sources.
def initial_checkpoint(args, config):
    from safetensors.torch import load_file
    path = args.out / 'initial.pt'
    if not path.exists():
        pretrained = legacy.download_pretrained(args.out / 'pretrained')
        base.seed_all(config['seed'])
        model = base.make_model()
        model[0].load_state_dict(load_file(str(pretrained)), strict=True)
        base.save_checkpoint(path, base.cpu_state(model), dict(config_sha256=shared.fingerprint(config)))
    saved = base.load_checkpoint(path)
    if saved['metadata']['config_sha256'] != shared.fingerprint(config):
        raise ValueError('Initialization belongs to another run configuration')
    return path


# Train each source on the manifest's 50-per-class sample, never on a held-out target.
def sources(args, manifest, config):
    initial = initial_checkpoint(args, config)
    identity = dict(initial_sha256=shared.file_hash(initial), manifest_sha256=manifest['sha256'])
    for domain in shared.DOMAINS:
        base.seed_all(config['seed'])
        model = base.make_model()
        shared.load_weights(model, initial)
        settings = dict(config['settings'], max_epochs=args.source_epochs)
        shared.train(model, args.data, manifest['domains'][domain], args.out / 'sources' / domain,
                     dict(identity, domain=domain), device=args.device, workers=args.workers,
                     seed=config['seed'], settings=settings, source=True)


# Average two newly trained source models, then independently adapt to each other domain.
def pairs(args, manifest, config):
    selected = [tuple(sorted(args.pair, key=shared.DOMAINS.index))] if args.pair else itertools.combinations(shared.DOMAINS, 2)
    for first, second in selected:
        source_paths = [args.out / 'sources' / d / 'final.pt' for d in (first, second)]
        for domain, path in zip((first, second), source_paths):
            result = json.loads(path.with_name('result.json').read_text())
            job = result['job']
            if (not job['source'] or job['identity']['manifest_sha256'] != manifest['sha256']
                    or job['identity']['domain'] != domain
                    or job['settings'] != dict(config['settings'], max_epochs=args.source_epochs)):
                raise ValueError(f'Unaligned source checkpoint: {path}')
            saved = base.load_checkpoint(path)
            if saved['metadata']['job_hash'] != shared.fingerprint(job):
                raise ValueError(f'Source checkpoint does not match source job: {path}')
        identity = dict(sources=[first, second], source_sha256=[shared.file_hash(p) for p in source_paths],
                        manifest_sha256=manifest['sha256'])
        directory = args.out / 'pairs' / f'{first}__{second}'
        directory.mkdir(parents=True, exist_ok=True)
        average = directory / 'average.pt'
        if average.exists():
            if base.load_checkpoint(average)['metadata'] != identity:
                raise ValueError('Source weights changed after averaging')
        else:
            states = [base.load_checkpoint(p)['state'] for p in source_paths]
            base.save_checkpoint(average, shared.average_states(*states), identity)
            del states
        for target in shared.DOMAINS:
            if target in (first, second):
                continue
            base.seed_all(config['seed'])
            model = base.make_model()
            shared.load_weights(model, average)
            shared.train(model, args.data, manifest['domains'][target], directory / target,
                         dict(identity, target=target, average_sha256=shared.file_hash(average)),
                         device=args.device, workers=args.workers, seed=config['seed'], settings=config['settings'])
            del model
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache()


# Apply exactly the same Phase 2 loop to a fine-tuning or KD initialization from another branch.
def adapt(args, manifest, config):
    if args.init is None or args.target is None:
        raise ValueError('adapt requires --init CHECKPOINT and --target DOMAIN')
    base.seed_all(config['seed'])
    model = base.make_model()
    shared.load_weights(model, args.init)
    identity = dict(init_sha256=shared.file_hash(args.init), target=args.target,
                    manifest_sha256=manifest['sha256'])
    shared.train(model, args.data, manifest['domains'][args.target], args.out / 'adapt' / args.target,
                 identity, device=args.device, workers=args.workers,
                 seed=config['seed'], settings=config['settings'])


# Export optimizer-step convergence, plateau status and sample counts without NaN placeholders.
def report(args):
    rows = []
    for path in sorted((args.out / 'pairs').glob('*/*/result.json')):
        result = json.loads(path.read_text())
        identity = result['job']['identity']
        rows.append(dict(sources='+'.join(identity['sources']), target=identity['target'],
                         steps_to_95pct_best=result['converged_at'], total_steps=result['steps'],
                         epochs_to_95pct_best=result['converged_at'] / result['steps_per_epoch'],
                         best_accuracy=result['best_accuracy'], stopped_by=result['stopped_by'],
                         budget_censored=result['budget_censored'], n_train=result['n_train'], n_test=result['n_test']))
    if rows:
        path = args.out / 'aligned_pair_convergence.csv'
        temporary = path.with_suffix('.csv.tmp')
        with temporary.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    print(f'Completed {len(rows)}/60 target adaptations', flush=True)


# Keep protocol settings explicit and make every long stage independently resumable.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'manifest', 'sources', 'pairs', 'adapt', 'report', 'all'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--source-epochs', type=int, default=10)
    parser.add_argument('--precision', choices=['float32', 'bfloat16', 'float16'], default='bfloat16')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--pair', nargs=2, choices=shared.DOMAINS)
    parser.add_argument('--target', choices=shared.DOMAINS)
    parser.add_argument('--init', type=Path)
    args = parser.parse_args()
    if args.source_epochs < 1 or args.workers < 0:
        parser.error('source-epochs must be positive and workers non-negative')
    if args.pair and args.pair[0] == args.pair[1]:
        parser.error('Choose two distinct sources')
    if args.stage == 'report':
        report(args)
        return
    if args.stage in ('prepare', 'all'):
        dino_data.prepare_dataset(args.data)
    manifest = (shared.load_manifest(args.data, args.manifest) if args.manifest.exists()
                else shared.make_manifest(args.data, args.manifest, args.seed))
    if args.seed != manifest['seed']:
        parser.error('--seed must match the shared manifest')
    config = configure(args, manifest)
    if args.stage in ('prepare', 'manifest'):
        for domain, membership in manifest['domains'].items():
            print(domain, 'train:', len(membership['train']), 'test:', len(membership['test']),
                  'classes below 50:', len(membership['classes_below_50']))
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('Select a GPU runtime or use --device cpu --precision float32')
    if args.stage in ('sources', 'all'):
        sources(args, manifest, config)
    if args.stage in ('pairs', 'all'):
        pairs(args, manifest, config)
    if args.stage == 'adapt':
        adapt(args, manifest, config)
    report(args)


if __name__ == '__main__':
    main()
