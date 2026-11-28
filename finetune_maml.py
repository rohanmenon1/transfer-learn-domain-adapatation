
# finetune_maml.py

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    DOMAINS,
    NUM_CLASSES,
    DomainNet,
    accuracy,
    backbone,
    test_loader,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune MAML initialization on target DomainNet domains"
    )
    p.add_argument("--experiment", default="default")
    p.add_argument("--backbone", default="dinov3")
    p.add_argument(
        "--maml-checkpoint",
        default="models/maml_domainnet.pt",
    )
    p.add_argument(
        "--targets",
        nargs="+",
        default=["clipart", "infograph", "quickdraw", "sketch"],
    )

    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=100, help="Evaluate test accuracy every N optimizer steps")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args()


def load_maml_initialization(model, checkpoint):
    """
    Transfer compatible MAML parameters into the 345-class model.

    The MAML classifier has n_way outputs (normally 5),
    whereas the fine-tuning classifier has 345 outputs.

    Consequently, the classifier is reinitialized, while
    the compatible feature extractor parameters are retained.
    """

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    current = model.state_dict()

    compatible = {
        key: value
        for key, value in state.items()
        if key in current
        and current[key].shape == value.shape
    }

    if not compatible:
        raise RuntimeError(
            "No compatible parameters found in MAML checkpoint."
        )

    missing, unexpected = model.load_state_dict(
        compatible,
        strict=False,
    )

    print(
        f"Loaded {len(compatible)}/{len(current)} parameter tensors "
        "from MAML initialization."
    )

    return compatible


def train_one_epoch(model, loader, optimizer, scheduler):
    if torch.cuda.is_available():
        print(
            f"Allocated: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB | "
            f"Reserved: "
            f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB | "
            f"Peak: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB",
            flush=True,
        )
        print(torch.cuda.mem_get_info())

    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in loader:
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += y.size(0)

    return {
        "train_loss": total_loss / total_samples,
        "train_accuracy": total_correct / total_samples,
    }


def fine_tune_domain(args, domain, maml_checkpoint):
    print(f"\n{'=' * 60}")
    print(f"Fine-tuning on {domain}")
    print(f"{'=' * 60}")

    # Free any VRAM left over from a previous domain before allocating
    # the new model + dataloader workers.  This prevents the CUBLAS
    # execution failures that surface when memory is fragmented.
    torch.cuda.empty_cache()

    torch.manual_seed(args.seed)

    # Construct a standard 345-class classifier.
    model = backbone(args.backbone, NUM_CLASSES)

    # Load the MAML feature extractor initialization.
    load_maml_initialization(model, maml_checkpoint)

    # Target-domain training data.
    train = DataLoader(
        DomainNet(
            domain,
            "train",
            train=True,
            per_class=args.per_class,
            seed=args.seed,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
    )

    # Target-domain evaluation data.
    test = test_loader(
        domain,
        limit=2000,
        seed=args.seed,
    )

    # Separate learning rates for backbone and classifier.
    head = list(model.get_classifier().parameters())
    head_ids = {id(p) for p in head}

    body = [
        p for p in model.parameters()
        if id(p) not in head_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": body, "lr": args.lr},
            {"params": head, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    warmup_steps = len(train)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1.0),
    )

    global_step = 0
    history = []

    # Evaluate the initialized model before fine-tuning.
    initial_accuracy = accuracy(model, test)
    torch.cuda.empty_cache()

    print(f"Initial accuracy (step 0): {initial_accuracy:.4f}")
    history.append({
        "step": 0,
        "epoch": 0,
        "train_loss": None,
        "train_accuracy": None,
        "test_accuracy": initial_accuracy,
    })

    for epoch in range(1, args.epochs + 1):
        torch.cuda.reset_peak_memory_stats()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for x, y in train:
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.float(), y)

            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1

            total_loss += loss.item() * y.size(0)
            total_correct += (logits.argmax(1) == y).sum().item()
            total_samples += y.size(0)

            # Step-level evaluation if eval_every is specified
            if args.eval_every and (global_step % args.eval_every == 0):
                test_accuracy = accuracy(model, test)
                torch.cuda.empty_cache()
                model.train()

                record = {
                    "step": global_step,
                    "epoch": epoch,
                    "train_loss": total_loss / total_samples,
                    "train_accuracy": total_correct / total_samples,
                    "test_accuracy": test_accuracy,
                }
                history.append(record)

                print(
                    f"Step {global_step:4d} (Epoch {epoch:2d}/{args.epochs}) | "
                    f"loss {record['train_loss']:.4f} | "
                    f"train acc {record['train_accuracy']:.4f} | "
                    f"test acc {test_accuracy:.4f}",
                    flush=True,
                )

        # End of epoch evaluation if not just evaluated at this exact step
        if not history or history[-1]["step"] != global_step:
            test_accuracy = accuracy(model, test)
            torch.cuda.empty_cache()

            record = {
                "step": global_step,
                "epoch": epoch,
                "train_loss": total_loss / total_samples,
                "train_accuracy": total_correct / total_samples,
                "test_accuracy": test_accuracy,
            }
            history.append(record)

            print(
                f"Epoch {epoch:3d}/{args.epochs} (Step {global_step:4d}) | "
                f"loss {record['train_loss']:.4f} | "
                f"train acc {record['train_accuracy']:.4f} | "
                f"test acc {test_accuracy:.4f}",
                flush=True,
            )

    # Save model and metrics independently for each domain.
    output_name = (
        f"maml_ft_{args.experiment}_{domain}"
        f"_{args.backbone}_pc{args.per_class}_s{args.seed}"
    )       

    Path("models").mkdir(exist_ok=True)
    Path("results").mkdir(exist_ok=True)

    torch.save(
        model.state_dict(),
        f"models/{output_name}.pt",
    )

    peak_accuracy = max(h["test_accuracy"] for h in history)
    thresh_95 = 0.95 * peak_accuracy
    converged_at_step = next(
        (h["step"] for h in history if h["test_accuracy"] >= thresh_95),
        None,
    )

    result = {
        "args": vars(args),
        "domain": domain,
        "steps_per_epoch": len(train),
        "initial_accuracy": initial_accuracy,
        "final_accuracy": history[-1]["test_accuracy"],
        "peak_accuracy": peak_accuracy,
        "converged_at_step": converged_at_step,
        "history": history,
    }

    Path(f"results/{output_name}.json").write_text(
        json.dumps(result, indent=2)
    )

    print(
        f"{domain}: {initial_accuracy:.4f} -> "
        f"{history[-1]['test_accuracy']:.4f}"
    )

    return result


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    if not Path(args.maml_checkpoint).exists():
        raise FileNotFoundError(args.maml_checkpoint)

    if len(set(args.targets)) != len(args.targets):
        raise ValueError("Target domains must be distinct.")

    if not set(args.targets).issubset(DOMAINS):
        raise ValueError(f"Invalid target domains: {args.targets}")

    maml_checkpoint = args.maml_checkpoint

    results = {}

    # Each target starts from the SAME MAML initialization.
    # Fine-tuning one target must not affect another.
    for domain in args.targets:
        results[domain] = fine_tune_domain(
            args,
            domain,
            maml_checkpoint,
        )

    summary = {
        domain: {
            "initial_accuracy": result["initial_accuracy"],
            "final_accuracy": result["final_accuracy"],
        }
        for domain, result in results.items()
    }

    print("\nFinal results:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
