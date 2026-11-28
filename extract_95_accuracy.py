#!/usr/bin/env python3
"""
extract_95_accuracy.py

Extracts 95% of the maximum accuracy and calculates the number of epochs
until 95% of peak accuracy is reached for each fine-tuned result trained
on 2 domains (DomainNet pairs).

Usage:
    python extract_95_accuracy.py
    python extract_95_accuracy.py --threshold 0.95 --results-dir results
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

STEPS_PER_EPOCH = {
    "clipart": 248,
    "infograph": 230,
    "painting": 237,
    "quickdraw": 269,
    "real": 268,
    "sketch": 255,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 95% of peak accuracy and epochs to reach it for 2-domain fine-tuning results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to the results directory (default: results)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Fraction of peak accuracy to reach (default: 0.95)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/epochs_to_95_peak.csv"),
        help="Path for saving CSV summary (default: results/epochs_to_95_peak.csv)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/epochs_to_95_peak.json"),
        help="Path for saving JSON summary (default: results/epochs_to_95_peak.json)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Do not print the detailed table to stdout",
    )
    return parser.parse_args()


def process_result_file(file_path: Path, threshold_fraction: float) -> Optional[Dict[str, Any]]:
    """
    Parses a single fine-tuning result JSON file and extracts accuracy metrics.
    Only processes experiments trained on 2 source domains.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to read {file_path}: {e}")
        return None

    args = data.get("args", {})
    experiment = args.get("experiment", "")

    # Experiments trained on 2 domains have experiment format "domain1_domain2"
    source_parts = experiment.split("_")
    if len(source_parts) != 2:
        return None

    source_1, source_2 = source_parts
    target = data.get("domain", "")
    backbone = args.get("backbone", "")
    per_class = args.get("per_class", None)
    seed = args.get("seed", None)

    initial_acc = data.get("initial_accuracy", 0.0)
    final_acc = data.get("final_accuracy", 0.0)
    history = data.get("history", [])

    if not history:
        return None

    # Track peak accuracy and peak epoch
    peak_acc = -1.0
    peak_epoch = -1

    for entry in history:
        test_acc = entry.get("test_accuracy", 0.0)
        epoch = entry.get("epoch", 0)
        if test_acc > peak_acc:
            peak_acc = test_acc
            peak_epoch = epoch

    if peak_acc <= 0:
        return None

    # Target threshold is 95% of peak accuracy
    threshold_value = threshold_fraction * peak_acc

    # Determine the number of epochs to reach the threshold
    epochs_to_threshold: Optional[int] = None
    acc_at_threshold_epoch: Optional[float] = None

    if initial_acc >= threshold_value:
        epochs_to_threshold = 0
        acc_at_threshold_epoch = initial_acc
    else:
        for entry in sorted(history, key=lambda x: x.get("epoch", 0)):
            test_acc = entry.get("test_accuracy", 0.0)
            epoch = entry.get("epoch", 0)
            if test_acc >= threshold_value:
                epochs_to_threshold = epoch
                acc_at_threshold_epoch = test_acc
                break

    # Calculate continuous interpolated step to 95% threshold
    steps_per_ep = data.get("steps_per_epoch", STEPS_PER_EPOCH.get(target, 250))
    pts = [(0.0, initial_acc)] + [
        (float(entry.get("step", entry.get("epoch", 0) * steps_per_ep)), float(entry.get("test_accuracy", 0.0)))
        for entry in sorted(history, key=lambda x: x.get("epoch", 0))
    ]
    interp_step: Optional[float] = None
    for i in range(len(pts) - 1):
        x1, y1 = pts[i]
        x2, y2 = pts[i + 1]
        if y1 <= threshold_value <= y2 and y2 > y1:
            frac = (threshold_value - y1) / (y2 - y1)
            interp_step = round(x1 + frac * (x2 - x1), 1)
            break
        elif y1 >= threshold_value:
            interp_step = x1
            break

    # Also compute 95% of total improvement (from initial to peak) as additional context
    improvement = peak_acc - initial_acc
    improvement_95_threshold = initial_acc + (threshold_fraction * improvement)
    epochs_to_95_gain: Optional[int] = None
    for entry in sorted(history, key=lambda x: x.get("epoch", 0)):
        test_acc = entry.get("test_accuracy", 0.0)
        epoch = entry.get("epoch", 0)
        if test_acc >= improvement_95_threshold:
            epochs_to_95_gain = epoch
            break

    return {
        "file": file_path.name,
        "source_1": source_1,
        "source_2": source_2,
        "target": target,
        "backbone": backbone,
        "per_class": per_class,
        "seed": seed,
        "initial_accuracy": initial_acc,
        "peak_accuracy": peak_acc,
        "peak_epoch": peak_epoch,
        "threshold_fraction": threshold_fraction,
        "target_95_accuracy": threshold_value,
        "epochs_to_95_peak": epochs_to_threshold,
        "steps_to_95_peak": interp_step,
        "steps_per_epoch": steps_per_ep,
        "accuracy_at_95_epoch": acc_at_threshold_epoch,
        "final_accuracy": final_acc,
        "total_improvement": peak_acc - initial_acc,
        "epochs_to_95_improvement": epochs_to_95_gain,
    }


def main():
    args = parse_args()
    results_dir = args.results_dir

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory does not exist: {results_dir}")

    # Gather all candidate json files (maml_ft_*.json)
    json_files = sorted(results_dir.glob("maml_ft_*.json"))
    if not json_files:
        # Fallback to scanning all json files in results dir
        json_files = sorted(results_dir.glob("*.json"))

    records: List[Dict[str, Any]] = []
    for f in json_files:
        rec = process_result_file(f, args.threshold)
        if rec is not None:
            records.append(rec)

    # Sort results by source_1, source_2, target
    records.sort(key=lambda r: (r["source_1"], r["source_2"], r["target"]))

    if not records:
        print(f"No 2-domain fine-tuned results found in {results_dir}.")
        return

    # Print summary table if not quiet
    if not args.quiet:
        pct_label = f"{args.threshold * 100:.0f}%"
        print("=" * 115)
        print(
            f"{'Source 1':<12} {'Source 2':<12} {'Target':<12} "
            f"{'Init Acc':<10} {'Peak Acc':<10} {'Peak Ep':<8} "
            f"{f'{pct_label} Peak Thr':<13} {f'Epochs to {pct_label}':<14} "
            f"{'Acc at Ep':<10} {'Final Acc':<10}"
        )
        print("-" * 115)

        epochs_list = []
        for r in records:
            ep_str = str(r["epochs_to_95_peak"]) if r["epochs_to_95_peak"] is not None else "N/A"
            acc_at_ep_str = f"{r['accuracy_at_95_epoch']:.4f}" if r["accuracy_at_95_epoch"] is not None else "N/A"

            if r["epochs_to_95_peak"] is not None:
                epochs_list.append(r["epochs_to_95_peak"])

            print(
                f"{r['source_1']:<12} {r['source_2']:<12} {r['target']:<12} "
                f"{r['initial_accuracy']:<10.4f} {r['peak_accuracy']:<10.4f} {r['peak_epoch']:<8d} "
                f"{r['target_95_accuracy']:<13.4f} {ep_str:<14} "
                f"{acc_at_ep_str:<10} {r['final_accuracy']:<10.4f}"
            )

        print("=" * 115)
        print(f"Total experiments evaluated: {len(records)}")
        if epochs_list:
            avg_epochs = sum(epochs_list) / len(epochs_list)
            print(f"Average epochs to reach {pct_label} of peak accuracy: {avg_epochs:.2f} epochs")

        # Group by target domain for average epochs
        targets = sorted(set(r["target"] for r in records))
        print("\nTarget domain breakdown:")
        print(f"{'Target':<14} {'Mean Epochs to ' + pct_label:<25} {'Mean Peak Acc':<15}")
        print("-" * 55)
        for t in targets:
            t_epochs = [r["epochs_to_95_peak"] for r in records if r["target"] == t and r["epochs_to_95_peak"] is not None]
            t_peak = [r["peak_accuracy"] for r in records if r["target"] == t]
            mean_ep = f"{sum(t_epochs) / len(t_epochs):.2f}" if t_epochs else "N/A"
            mean_pk = f"{sum(t_peak) / len(t_peak):.4f}" if t_peak else "N/A"
            print(f"{t:<14} {mean_ep:<25} {mean_pk:<15}")

    # Write CSV
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "source_1",
            "source_2",
            "target",
            "backbone",
            "initial_accuracy",
            "peak_accuracy",
            "peak_epoch",
            "threshold_fraction",
            "target_95_accuracy",
            "epochs_to_95_peak",
            "steps_to_95_peak",
            "accuracy_at_95_epoch",
            "final_accuracy",
            "total_improvement",
            "epochs_to_95_improvement",
            "file",
        ]
        with args.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in records:
                writer.writerow(r)
        print(f"\nSaved CSV results to: {args.output_csv}")

    # Write JSON
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)
        print(f"Saved JSON results to: {args.output_json}")


if __name__ == "__main__":
    main()
