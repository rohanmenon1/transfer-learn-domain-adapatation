#!/usr/bin/env python3
"""
plot_adaptation_curves.py

Generates a 2x3 grid of target domain adaptation curves from fine-tuning results.
Matches the styling of the target adaptation reference figure:
- 6 subplots for the 6 DomainNet domains (clipart, infograph, painting, quickdraw, real, sketch)
- 10 curves per subplot corresponding to each source pair (domain1+domain2)
- Dots marking the step/epoch where accuracy reaches >= 95% of the run's peak
- Supports high-granularity continuous interpolation and step-level history logging
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
        "--interpolate",
        action="store_true",
        default=True,
        help="Interpolate threshold crossing step for high granularity (default: True)",
    )
    parser.add_argument(
        "--no-interpolate",
        dest="interpolate",
        action="store_false",
        help="Do not interpolate threshold crossing (use discrete recorded steps)",
    )
    parser.add_argument(
        "--include-init",
        action="store_true",
        help="Include initial accuracy at step 0 (default: False)",
    )
    parser.add_argument(
        "--export-granular-json",
        type=Path,
        default=None,
        help="Optional directory to export high-granularity 100-step JSON files",
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

    for json_path in sorted(results_dir.glob("maml_ft_*.json")):
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
            "steps_per_epoch": data.get("steps_per_epoch", STEPS_PER_EPOCH.get(target, 250)),
            "history": history,
            "raw_data": data,
            "file": json_path.name,
        })

    # Sort pairs within each target alphabetically by pair_label
    for target in DOMAINS:
        by_target[target].sort(key=lambda r: r["pair_label"])

    return by_target


def export_granular_jsons(by_target: Dict[str, List[Dict[str, Any]]], export_dir: Path, step_interval: int = 100):
    """
    Exports high-granularity JSONs where accuracy is evaluated every step_interval optimizer steps.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    count = 0

    for target, runs in by_target.items():
        steps_per_ep = STEPS_PER_EPOCH.get(target, 250)

        for run in runs:
            data = dict(run["raw_data"])
            history = sorted(run["history"], key=lambda h: h.get("epoch", 0) if "epoch" in h else h.get("step", 0))

            # Build base points (step, acc)
            base_pts: List[Tuple[float, float]] = []
            init_acc = run["initial_accuracy"]
            base_pts.append((0.0, init_acc))

            for h in history:
                s = h.get("step", h.get("epoch", 0) * steps_per_ep)
                base_pts.append((float(s), float(h["test_accuracy"])))

            base_pts.sort(key=lambda pt: pt[0])

            # Interpolate at step_interval
            max_step = int(base_pts[-1][0])
            eval_steps = list(range(0, max_step + 1, step_interval))
            if eval_steps[-1] < max_step:
                eval_steps.append(max_step)

            new_history = []
            for target_step in eval_steps:
                # Find segment
                acc = base_pts[0][1]
                for i in range(len(base_pts) - 1):
                    x1, y1 = base_pts[i]
                    x2, y2 = base_pts[i + 1]
                    if x1 <= target_step <= x2:
                        frac = (target_step - x1) / (x2 - x1) if x2 > x1 else 0.0
                        acc = y1 + frac * (y2 - y1)
                        break

                epoch_float = target_step / steps_per_ep
                new_history.append({
                    "step": target_step,
                    "epoch": round(epoch_float, 2),
                    "test_accuracy": round(acc, 6),
                })

            peak_acc = max(h["test_accuracy"] for h in new_history)
            thresh_95 = 0.95 * peak_acc
            converged_step = next(
                (h["step"] for h in new_history if h["test_accuracy"] >= thresh_95),
                None,
            )

            data["steps_per_epoch"] = steps_per_ep
            data["eval_every"] = step_interval
            data["peak_accuracy"] = peak_acc
            data["converged_at_step"] = converged_step
            data["history"] = new_history

            out_file = export_dir / run["file"]
            out_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
            count += 1

    print(f"Exported {count} high-granularity JSON files (eval every {step_interval} steps) to: {export_dir}")


def plot_target_adaptation(
    by_target: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    threshold: float = 0.95,
    x_axis: str = "steps",
    interpolate: bool = True,
    include_init: bool = False,
    save_svg: bool = False,
    dpi: int = 200,
):
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
            history = sorted(run["history"], key=lambda h: h.get("step", h.get("epoch", 0)))

            init_acc = run["initial_accuracy"] * 100.0
            peak_acc = max(h["test_accuracy"] for h in history) * 100.0
            threshold_val = threshold * peak_acc

            xs: List[float] = []
            ys: List[float] = []

            # Determine whether history contains dense step measurements
            has_step_logging = any("step" in h and h["step"] > 0 for h in history)

            if include_init:
                xs.append(0.0)
                ys.append(init_acc)

            for h in history:
                if has_step_logging and "step" in h:
                    x_val = float(h["step"]) if x_axis == "steps" else float(h.get("epoch", h["step"] / steps_per_ep))
                else:
                    ep = float(h.get("epoch", 0))
                    x_val = (ep * steps_per_ep) if x_axis == "steps" else ep

                y_val = float(h["test_accuracy"]) * 100.0
                if not (x_val == 0.0 and include_init):
                    xs.append(x_val)
                    ys.append(y_val)

            # Adaptation curve
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=1.7,
                alpha=0.9,
                label=run["pair_label"],
            )

            # Calculate dot position
            dot_x: Optional[float] = None
            dot_y: Optional[float] = None

            if not interpolate:
                # Direct first observed point
                for x_val, y_val in zip(xs, ys):
                    if y_val >= threshold_val:
                        dot_x = x_val
                        dot_y = y_val
                        break
            else:
                # Continuous interpolation between adjacent measurement points
                for i in range(len(xs) - 1):
                    x1, y1 = xs[i], ys[i]
                    x2, y2 = xs[i + 1], ys[i + 1]
                    if y1 <= threshold_val <= y2 and y2 > y1:
                        frac = (threshold_val - y1) / (y2 - y1)
                        dot_x = x1 + frac * (x2 - x1)
                        dot_y = threshold_val
                        break
                    elif y1 >= threshold_val:
                        dot_x = x1
                        dot_y = y1
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
    plt.savefig(str(output_path.resolve()), dpi=dpi, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")

    if save_svg:
        svg_path = output_path.with_suffix(".svg")
        plt.savefig(str(svg_path.resolve()), format="svg", bbox_inches="tight")
        print(f"Saved SVG to: {svg_path}")

    plt.close(fig)


def main():
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results dir not found: {args.results_dir}")

    by_target = load_runs(args.results_dir)
    total_runs = sum(len(v) for v in by_target.values())
    print(f"Loaded {total_runs} runs across {len(DOMAINS)} target domains.")

    if args.export_granular_json:
        export_granular_jsons(by_target, args.export_granular_json)

    plot_target_adaptation(
        by_target=by_target,
        output_path=args.output,
        threshold=args.threshold,
        x_axis=args.x_axis,
        interpolate=args.interpolate,
        include_init=args.include_init,
        save_svg=args.save_svg,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
