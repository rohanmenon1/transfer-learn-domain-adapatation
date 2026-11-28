#!/usr/bin/env python3
"""
plot_adaptation_curves.py

Generates a 2x3 grid of target domain adaptation curves from fine-tuning results.
Matches the styling of the target adaptation reference figure:
- 6 subplots for the 6 DomainNet domains (clipart, infograph, painting, quickdraw, real, sketch)
- 10 curves per subplot corresponding to each source pair (domain1+domain2)
- Dots marking the first observed step/epoch where accuracy >= 95% of the run's peak
- Clean layout with accurate axis limits, grid, legends, and suptitle
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]

# Number of batches (optimizer steps) per epoch for DomainNet train split (per_class=50, batch_size=64)
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
        description="Plot target domain adaptation curves matching reference figure."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Path to results directory containing maml_ft_*.json files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/target_adaptation_curves.png"),
        help="Output image path (default: results/target_adaptation_curves.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Fraction of peak accuracy to mark with a dot (default: 0.95)",
    )
    parser.add_argument(
        "--x-axis",
        choices=["steps", "epochs"],
        default="steps",
        help="Metric for x-axis: 'steps' (optimizer steps) or 'epochs' (default: steps)",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Include initial accuracy at step 0 (default: False)",
    )
    parser.add_argument(
        "--save-svg",
        action="store_true",
        help="Also save an SVG version of the plot",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI resolution for raster output (default: 200)",
    )
    return parser.parse_args()


def load_runs(results_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads all fine-tuning result files and groups them by target domain.
    """
    by_target: Dict[str, List[Dict[str, Any]]] = {d: [] for d in DOMAINS}

    for json_path in results_dir.glob("maml_ft_*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not read {json_path}: {e}")
            continue

        target = data.get("domain")
        if target not in by_target:
            continue

        args = data.get("args", {})
        experiment = args.get("experiment", "")
        parts = experiment.split("_")
        if len(parts) != 2:
            continue

        source_1, source_2 = parts
        history = data.get("history", [])
        if not history:
            continue

        by_target[target].append({
            "source_1": source_1,
            "source_2": source_2,
            "pair_label": f"{source_1}+{source_2}",
            "initial_accuracy": data.get("initial_accuracy", 0.0),
            "final_accuracy": data.get("final_accuracy", 0.0),
            "history": history,
            "file": json_path.name,
        })

    # Sort pairs within each target alphabetically by pair_label
    for target in DOMAINS:
        by_target[target].sort(key=lambda r: r["pair_label"])

    return by_target


def plot_target_adaptation(
    by_target: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    threshold: float = 0.95,
    x_axis: str = "steps",
    include_init: bool = False,
    save_svg: bool = False,
    dpi: int = 200,
):
    # Set up matplotlib style matching reference
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    })

    fig, axes = plt.subplots(2, 3, figsize=(17, 9.2), sharey=True)
    pct_label = f"{threshold * 100:.0f}%" if threshold == int(threshold * 100) / 100 else f"{threshold * 100:.1f}%"
    fig.suptitle(
        f"Target adaptation: dots mark first observed {x_axis[:-1] if x_axis == 'steps' else 'epoch'} ≥{pct_label} of run best",
        fontsize=13,
        fontweight="normal",
        y=0.985,
    )

    tab10_colors = plt.get_cmap("tab10").colors

    for idx, target in enumerate(DOMAINS):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]

        runs = by_target.get(target, [])
        steps_per_ep = STEPS_PER_EPOCH.get(target, 250)

        for p_idx, run in enumerate(runs):
            color = tab10_colors[p_idx % len(tab10_colors)]
            history = sorted(run["history"], key=lambda h: h.get("epoch", 0))

            # Accuracies in percentage
            init_acc = run["initial_accuracy"] * 100.0
            peak_acc = max(h["test_accuracy"] for h in history) * 100.0
            threshold_val = threshold * peak_acc

            xs: List[float] = []
            ys: List[float] = []

            if include_init:
                xs.append(0)
                ys.append(init_acc)

            for h in history:
                ep = h["epoch"]
                x_val = (ep * steps_per_ep) if x_axis == "steps" else ep
                y_val = h["test_accuracy"] * 100.0
                xs.append(x_val)
                ys.append(y_val)

            # Plot adaptation curve
            line = ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.7,
                alpha=0.9,
                label=run["pair_label"],
            )

            # Find first point >= threshold
            dot_x: Optional[float] = None
            dot_y: Optional[float] = None
            for x_val, y_val in zip(xs, ys):
                if y_val >= threshold_val:
                    dot_x = x_val
                    dot_y = y_val
                    break

            if dot_x is not None and dot_y is not None:
                ax.scatter(
                    [dot_x],
                    [dot_y],
                    color=color,
                    s=32,
                    zorder=5,
                    edgecolors="none",
                )

        ax.set_title(target, fontsize=12, pad=6)
        ax.set_ylabel("Monitoring accuracy (%)", fontsize=10.5)
        ax.set_xlabel("Optimizer steps" if x_axis == "steps" else "Epochs", fontsize=10.5)

        # Consistent axes ranges & grid
        ax.set_ylim(4, 86)
        ax.set_yticks([10, 20, 30, 40, 50, 60, 70, 80])

        if x_axis == "steps":
            ax.set_xlim(-60, 2800)
            ax.set_xticks([0, 500, 1000, 1500, 2000, 2500])
        else:
            ax.set_xlim(0.5 if not include_init else -0.5, 10.5)
            ax.set_xticks(list(range(0 if include_init else 1, 11)))

        ax.grid(True, linestyle="-", color="#eaeaea", linewidth=0.7, zorder=0)
        ax.set_axisbelow(True)

        # Legend positioning matching reference image
        leg_loc = "lower right" if target == "quickdraw" else "upper right"
        ax.legend(
            fontsize=6.5,
            loc=leg_loc,
            frameon=True,
            framealpha=0.88,
            edgecolor="#e0e0e0",
            borderpad=0.35,
            labelspacing=0.22,
            handlelength=1.4,
        )

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.97])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    if save_svg:
        svg_path = output_path.with_suffix(".svg")
        plt.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Saved SVG to: {svg_path}")

    plt.close(fig)


def main():
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    by_target = load_runs(args.results_dir)
    total_runs = sum(len(v) for v in by_target.values())
    print(f"Loaded {total_runs} runs across {len(DOMAINS)} target domains.")

    plot_target_adaptation(
        by_target=by_target,
        output_path=args.output,
        threshold=args.threshold,
        x_axis=args.x_axis,
        include_init=args.include_init,
        save_svg=args.save_svg,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
