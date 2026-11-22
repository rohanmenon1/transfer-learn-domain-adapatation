
# run_all_pairs.py

import itertools
import subprocess
import sys
import json
import csv

from pathlib import Path

from common import DOMAINS


# Experiment configuration
BACKBONE = "dinov3"

N_WAY = 5
K_SHOT = 5
Q_QUERY = 5

EPISODES = 550
META_LR = 1e-6
INNER_LR = 1e-3
INNER_STEPS = 1
META_BATCH = 2

PER_CLASS = 50
EPOCHS = 10
FT_LR = 3e-5
HEAD_LR = 1e-3

SEED = 0

MODEL_DIR = Path("models")
RESULT_DIR = Path("results")

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def run_command(command):
    """Execute a Python experiment and stop on failure."""

    print("\nRunning:", " ".join(command), flush=True)

    subprocess.run(
        command,
        check=True,
    )


def main():

    pairs = list(itertools.combinations(DOMAINS, 2))

    all_results = []

    print(f"Total source pairs: {len(pairs)}")
    print(f"Total experiments: {len(pairs) * 4}")

    for pair_idx, (source1, source2) in enumerate(pairs, 1):

        experiment = f"{source1}_{source2}"

        print("\n" + "=" * 70)
        print(
            f"PAIR {pair_idx}/{len(pairs)}: "
            f"{source1} + {source2}"
        )
        print("=" * 70)

        maml_checkpoint = MODEL_DIR / f"maml_{experiment}.pt"

        # -------------------------------------------------
        # 1. Train MAML on the two source domains.
        # -------------------------------------------------

        command = [
            sys.executable,
            "maml_domainnet.py",

            "--backbone", BACKBONE,

            "--domains", source1, source2,

            "--n-way", str(N_WAY),
            "--k-shot", str(K_SHOT),
            "--q-query", str(Q_QUERY),

            "--episodes", str(EPISODES),
            "--meta-lr", str(META_LR),
            "--inner-lr", str(INNER_LR),
            "--inner-steps", str(INNER_STEPS),
            "--meta-batch", str(META_BATCH),

            "--per-class", str(PER_CLASS),
            "--seed", str(SEED),

            "--output", f"maml_{experiment}",
        ]

        run_command(command)

        if not maml_checkpoint.exists():
            raise FileNotFoundError(maml_checkpoint)

        # -------------------------------------------------
        # 2. Fine-tune on each remaining target domain.
        # -------------------------------------------------

        targets = [
            d for d in DOMAINS
            if d not in (source1, source2)
        ]

        for target_idx, target in enumerate(targets, 1):

            print("\n" + "-" * 60)
            print(
                f"Fine-tuning {experiment} on {target} "
                f"({target_idx}/4)"
            )
            print("-" * 60)

            command = [
                sys.executable,
                "finetune_maml.py",

                "--backbone", BACKBONE,

                "--maml-checkpoint", str(maml_checkpoint),

                "--targets", target,

                "--per-class", str(PER_CLASS),
                "--epochs", str(EPOCHS),

                "--lr", str(FT_LR),
                "--head-lr", str(HEAD_LR),

                "--seed", str(SEED),

                "--experiment", experiment,
            ]

            run_command(command)

            # -------------------------------------------------
            # 3. Read the saved evaluation results.
            # -------------------------------------------------

            result_name = (
                f"maml_ft_{experiment}_{target}"
                f"_{BACKBONE}_pc{PER_CLASS}_s{SEED}"
            )

            result_path = RESULT_DIR / f"{result_name}.json"

            result = json.loads(result_path.read_text())

            record = {
                "source_1": source1,
                "source_2": source2,
                "target": target,
                "initial_accuracy": result["initial_accuracy"],
                "final_accuracy": result["final_accuracy"],
                "improvement": (
                    result["final_accuracy"]
                    - result["initial_accuracy"]
                ),
            }

            all_results.append(record)

            print("\nExperiment result:")
            print(json.dumps(record, indent=2))

            # Save after every experiment so progress is retained.
            save_results(all_results)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED")
    print("=" * 70)

    save_results(all_results)

    print_summary(all_results)


def save_results(results):

    json_path = RESULT_DIR / "all_pairs_results.json"
    csv_path = RESULT_DIR / "all_pairs_results.csv"

    json_path.write_text(
        json.dumps(results, indent=2)
    )

    if results:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=results[0].keys(),
            )
            writer.writeheader()
            writer.writerows(results)


def print_summary(results):

    print("\nResults by target domain:")

    for target in DOMAINS:

        rows = [
            r for r in results
            if r["target"] == target
        ]

        if not rows:
            continue

        mean_initial = sum(
            r["initial_accuracy"] for r in rows
        ) / len(rows)

        mean_final = sum(
            r["final_accuracy"] for r in rows
        ) / len(rows)

        print(
            f"{target:12s} | "
            f"Initial: {mean_initial:.4f} | "
            f"Final: {mean_final:.4f} | "
            f"Improvement: {mean_final - mean_initial:+.4f}"
        )

    print("\nOverall:")

    print(
        "Mean final accuracy:",
        sum(r["final_accuracy"] for r in results)
        / len(results),
    )


if __name__ == "__main__":
    main()
