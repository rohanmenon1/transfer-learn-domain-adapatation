# Run from a terminal or a Colab cell using the same data/results paths each time.
# Example installation in Colab:
# %pip install timm==1.0.31 safetensors==0.8.0 pyarrow huggingface_hub
# Example run after mounting Drive:
# !python run_experiment.py all --data /content/domainnet_data --out /content/drive/MyDrive/domainnet_dinov3_pair_epochs_seed42
# Stages: prepare, sources, pairs, report, all. Model training requires a GPU by default.
# The stopping rule applies to future completed epochs. Delta < 0.005 stops;
# delta >= 0.005 continues. Falling accuracy also stops. The 50-epoch cap remains.
# Partial jobs restore their last completed epoch. Completed jobs are skipped.
# Legacy 90%-criterion results keep their recorded criterion in the summaries.

import argparse
from pathlib import Path

import torch

import dino_data
import dino_epochs as experiment


# Parse the selected stage and the same settings used by the notebook.
def main():
    parser = argparse.ArgumentParser(description='DomainNet DINOv3 pair-averaging experiment')
    parser.add_argument('stage', choices=['prepare', 'sources', 'pairs', 'report', 'all'])
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-source-epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone-lr', type=float, default=1e-5)
    parser.add_argument('--head-lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=.05)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pair', nargs=2, choices=experiment.DOMAINS,
                        help='Run one displayed source pair instead of all fifteen')
    options = parser.parse_args()
    args = experiment.config(options.data, options.out, seed=options.seed,
        max_source_epochs=options.max_source_epochs, batch_size=options.batch_size,
        workers=options.workers, backbone_lr=options.backbone_lr,
        head_lr=options.head_lr, weight_decay=options.weight_decay, device=options.device)
    if options.stage == 'report':
        experiment.report(args)
        return
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Select a GPU runtime or pass --device cpu.')
    pretrained = experiment.download_pretrained(args.out / 'pretrained')
    dino_data.prepare_dataset(args.data)
    splits = experiment.prepare(args, pretrained)
    if options.stage in ('sources', 'all'):
        experiment.source_runs(args, splits)
    if options.stage in ('pairs', 'all'):
        selected = None
        if options.pair:
            if options.pair[0] == options.pair[1]:
                parser.error('A source pair must contain two different domains')
            selected = [tuple(sorted(options.pair, key=experiment.DOMAINS.index))]
        experiment.pair_runs(args, splits, selected)


if __name__ == '__main__':
    main()
