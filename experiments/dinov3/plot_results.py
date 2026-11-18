# Plot saved DINOv3 histories and completed results without loading model weights.
# Usage: python plot_results.py --run-dir /content/drive/MyDrive/domainnet_dinov3_pair_epochs_seed42
# Figures are written to the run's charts directory as PNG and PDF files.

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DOMAINS = ('clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch')


# Read a completed atomic JSON snapshot and tolerate a file not yet being present.
def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


# Save exportable figures and release their plotting resources.
def save_figure(figure, directory, name):
    figure.tight_layout()
    for extension in ('png', 'pdf'):
        figure.savefig(directory / f'{name}.{extension}', dpi=180, bbox_inches='tight')
    plt.close(figure)
    print('Saved', directory / f'{name}.png')


# Label old threshold runs and the newer plateau rule separately in the plots.
def stopping_label(result):
    status = result.get('status', 'running')
    if status == 'budget_exhausted':
        return 'epoch cap'
    if status == 'converged' or (result.get('stopping_criterion') or '').startswith('validation_improvement'):
        return 'gain <0.5 pp'
    if status == 'reached_90':
        return '90% threshold'
    return status


# Plot standalone learning curves, validation gains, and final performance.
def plot_sources(run, output):
    jobs = []
    for domain in DOMAINS:
        directory = run / 'sources' / domain
        history = read_json(directory / 'history.json', [])
        result = read_json(directory / 'result.json', {})
        if history:
            jobs.append((domain, history, result))
    if not jobs:
        print('No source histories have been saved yet.')
        return

    figure, axis = plt.subplots(figsize=(10, 6))
    for domain, history, result in jobs:
        epochs = [row['epoch'] for row in history]
        accuracy = [100 * row['validation_accuracy'] for row in history]
        line, = axis.plot(epochs, accuracy, marker='.', label=f'{domain} ({stopping_label(result)})')
        if result:
            axis.scatter(epochs[-1], accuracy[-1], marker='s', color=line.get_color())
    axis.set(xlabel='Completed epoch', ylabel='Validation top-1 accuracy (%)',
             title='DINOv3 standalone validation learning curves', ylim=(0, 100))
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    save_figure(figure, output, 'source_validation_curves')

    figure, axis = plt.subplots(figsize=(10, 6))
    for domain, history, result in jobs:
        if len(history) >= 2:
            epochs = [row['epoch'] for row in history[1:]]
            accuracy = np.array([row['validation_accuracy'] for row in history])
            axis.plot(epochs, 100 * np.diff(accuracy), marker='.', label=domain)
    axis.axhline(.5, color='black', linestyle='--', label='0.5 percentage-point criterion')
    axis.axhline(0, color='gray', linewidth=.7)
    axis.set(xlabel='Completed epoch', ylabel='Validation gain over preceding epoch (percentage points)',
             title='DINOv3 standalone changes in validation accuracy')
    axis.grid(alpha=.25)
    axis.legend(fontsize=8)
    save_figure(figure, output, 'source_validation_gains')

    completed = [(d, h, r) for d, h, r in jobs if r]
    if not completed:
        return
    positions = np.arange(len(completed))
    figure, axis = plt.subplots(figsize=(10, 6))
    validation = [100 * r.get('best_validation_accuracy', max(row['validation_accuracy'] for row in h))
                  for _, h, r in completed]
    test = [100 * r['test_accuracy'] if r.get('test_accuracy') is not None else np.nan
            for _, _, r in completed]
    axis.bar(positions - .2, validation, .4, label='Best validation accuracy')
    axis.bar(positions + .2, test, .4, label='Test accuracy of selected checkpoint')
    axis.set(xticks=positions, xticklabels=[d for d, _, _ in completed],
             ylabel='Top-1 accuracy (%)', ylim=(0, 100), title='Completed standalone model performance')
    axis.legend(fontsize=8)
    save_figure(figure, output, 'source_final_accuracy')

    figure, axis = plt.subplots(figsize=(10, 6))
    epochs = [r.get('epochs_completed', h[-1]['epoch']) for _, h, r in completed]
    bars = axis.bar(positions, epochs)
    for bar, (_, _, result) in zip(bars, completed):
        axis.annotate(stopping_label(result), (bar.get_x() + bar.get_width()/2, bar.get_height()),
                      xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8)
    axis.set(xticks=positions, xticklabels=[d for d, _, _ in completed],
             ylabel='Completed epochs', title='Standalone stopping epochs and recorded stopping reasons')
    axis.margins(y=.2)
    save_figure(figure, output, 'source_stopping_epochs')


# Plot each target's available pair-adaptation curves with its standalone curve.
def plot_pairs(run, output):
    for target in DOMAINS:
        paths = sorted((run / 'pairs').glob(f'*/{target}/history.json'))
        if not paths:
            continue
        figure, axis = plt.subplots(figsize=(11, 7))
        reference = read_json(run / 'sources' / target / 'history.json', [])
        if reference:
            axis.plot([r['epoch'] for r in reference],
                      [100 * r['validation_accuracy'] for r in reference],
                      color='black', linewidth=2.5, linestyle='--', label='Standalone target')
        for path in paths:
            history = read_json(path, [])
            if not history:
                continue
            result = read_json(path.parent / 'result.json', {})
            label = path.parent.parent.name.replace('__', ' + ')
            label += f' ({stopping_label(result)})'
            axis.plot([r['epoch'] for r in history],
                      [100 * r['validation_accuracy'] for r in history], marker='.', label=label)
        axis.set(xlabel='Target fine-tuning epoch', ylabel='Validation top-1 accuracy (%)',
                 ylim=(0, 100), title=f'Pair-averaged initializations adapted to {target}')
        axis.grid(alpha=.25)
        axis.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        save_figure(figure, output, f'pair_convergence_{target}')


# Generate charts using only the saved JSON files in the selected run folder.
def main():
    parser = argparse.ArgumentParser(description='Plot DINOv3 source and pair convergence results')
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error('The run directory does not exist. Mount Drive and check the path.')
    output = args.out or args.run_dir / 'charts'
    output.mkdir(parents=True, exist_ok=True)
    plot_sources(args.run_dir, output)
    plot_pairs(args.run_dir, output)


if __name__ == '__main__':
    main()
