import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import random
import time

import higher
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.data import DataLoader, Dataset

from common import DOMAINS, DomainNet, backbone


class RemappedDataset(Dataset):
    """Top-level dataset class to remap selected classes to [0, n_way - 1]."""

    def __init__(self, dataset, selected_indices, label_map):
        self.dataset = dataset
        self.selected_indices = selected_indices
        self.label_map = label_map

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        x, original_label = self.dataset[self.selected_indices[idx]]
        return x, self.label_map[original_label]


class EpisodeSampler:
    """Samples N-way K-shot classification episodes for MAML training."""

    def __init__(self, dataset, n_way, k_shot, q_query, seed):
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.rng = random.Random(seed)

        self.by_class = defaultdict(list)
        for i, (_, label) in enumerate(dataset.items):
            self.by_class[label].append(i)

        needed = k_shot + q_query
        self.classes = [
            c for c, indices in self.by_class.items() if len(indices) >= needed
        ]

        if len(self.classes) < n_way:
            raise ValueError(
                f"Need {n_way} classes with at least {needed} images each; found {len(self.classes)}."
            )

    def sample(self, device):
        classes = self.rng.sample(self.classes, self.n_way)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, original_label in enumerate(classes):
            indices = self.rng.sample(
                self.by_class[original_label],
                self.k_shot + self.q_query,
            )
            for i in indices[: self.k_shot]:
                x, _ = self.dataset[i]
                support_x.append(x)
                support_y.append(new_label)

            for i in indices[self.k_shot :]:
                x, _ = self.dataset[i]
                query_x.append(x)
                query_y.append(new_label)

        return (
            torch.stack(support_x).to(device),
            torch.tensor(support_y, device=device),
            torch.stack(query_x).to(device),
            torch.tensor(query_y, device=device),
        )


def make_maml_tasks(domains, args, device):
    """Instantiates episode samplers for each source domain."""
    tasks = {}
    for i, domain in enumerate(domains):
        dataset = DomainNet(
            domain,
            "train",
            train=True,
            per_class=args.per_class,
            seed=args.seed,
        )
        tasks[domain] = EpisodeSampler(
            dataset,
            args.n_way,
            args.k_shot,
            args.q_query,
            args.seed + i + 1,
        )
    return tasks


def meta_train_maml(model, tasks, args, device):
    """Pre-trains model using MAML on the provided source tasks."""
    model.train()
    meta_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.meta_lr,
        weight_decay=0.01,
    )
    task_keys = list(tasks.keys())
    rng = random.Random(args.seed + 1000)

    for episode in range(1, args.meta_episodes + 1):
        meta_optimizer.zero_grad(set_to_none=True)
        meta_loss_value, meta_acc_value = 0.0, 0.0

        for _ in range(args.meta_batch):
            domain = rng.choice(task_keys)
            support_x, support_y, query_x, query_y = tasks[domain].sample(device)

            inner_optimizer = torch.optim.SGD(model.parameters(), lr=args.inner_lr)

            with sdpa_kernel(SDPBackend.MATH):
                with higher.innerloop_ctx(
                    model,
                    inner_optimizer,
                    copy_initial_weights=False,
                    track_higher_grads=True,
                ) as (fast_model, diff_optimizer):
                    for _ in range(args.inner_steps):
                        support_logits = fast_model(support_x)
                        support_loss = F.cross_entropy(support_logits.float(), support_y)
                        diff_optimizer.step(support_loss)

                    query_logits = fast_model(query_x)
                    query_loss = F.cross_entropy(query_logits.float(), query_y)
                    query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()

                    (query_loss / args.meta_batch).backward()
                    meta_loss_value += query_loss.item() / args.meta_batch
                    meta_acc_value += query_acc.item() / args.meta_batch

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_optimizer.step()

        if episode % 50 == 0 or episode == args.meta_episodes:
            print(
                f"  [MAML Pre-train] Episode {episode}/{args.meta_episodes} | Loss: {meta_loss_value:.4f} | Acc: {meta_acc_value:.4f}",
                flush=True,
            )

    return model


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluates loss and accuracy on target evaluation set."""
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / max(total, 1), total_correct / max(total, 1)


def evaluate_target_convergence(model, target_domain, args, device):
    """Measures adaptation speed and convergence rate to the target domain."""
    train_dataset = DomainNet(
        target_domain,
        "train",
        train=True,
        per_class=args.per_class,
        seed=args.seed,
    )
    test_dataset = DomainNet(
        target_domain,
        "test",
        train=False,
        seed=args.seed,
    )

    by_class = defaultdict(list)
    for i, (_, label) in enumerate(train_dataset.items):
        by_class[label].append(i)

    eligible = [label for label, ids in by_class.items() if len(ids) >= args.adapt_batch_size]
    if len(eligible) < args.n_way:
        raise ValueError(
            f"Domain '{target_domain}' has only {len(eligible)} classes with >= {args.adapt_batch_size} samples (need {args.n_way})."
        )

    selected = sorted(random.Random(args.seed).sample(eligible, args.n_way))
    label_map = {label: i for i, label in enumerate(selected)}

    train_indices = [i for i, (_, label) in enumerate(train_dataset.items) if label in label_map]
    test_indices = [i for i, (_, label) in enumerate(test_dataset.items) if label in label_map]

    train_loader = DataLoader(
        RemappedDataset(train_dataset, train_indices, label_map),
        batch_size=args.adapt_batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        RemappedDataset(test_dataset, test_indices, label_map),
        batch_size=args.adapt_batch_size,
        shuffle=False,
        num_workers=2,
    )

    adapt_model = copy.deepcopy(model).to(device)
    optimizer = torch.optim.AdamW(adapt_model.parameters(), lr=args.adapt_lr)

    init_loss, init_acc = evaluate(adapt_model, test_loader, device)
    print(f"  [Adaptation: {target_domain}] Step 0 | Loss: {init_loss:.4f} | Acc: {init_acc:.4f}")

    history = [{"step": 0, "val_loss": init_loss, "val_acc": init_acc}]
    global_step = 0
    step_to_target = None
    start_time = time.time()

    for epoch in range(args.adapt_epochs):
        adapt_model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(adapt_model(images), labels)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % args.eval_every_steps == 0:
                val_loss, val_acc = evaluate(adapt_model, test_loader, device)
                history.append({"step": global_step, "val_loss": val_loss, "val_acc": val_acc})
                print(
                    f"    Epoch {epoch+1} | Step {global_step:3d} | Loss: {val_loss:.4f} | Acc: {val_acc:.4f}"
                )
                if step_to_target is None and val_acc >= args.target_acc:
                    step_to_target = global_step
                    print(f"    --> Reached target accuracy ({args.target_acc:.1%}) at step {global_step}!")
                adapt_model.train()

    elapsed = time.time() - start_time
    return {
        "step_to_target": step_to_target,
        "final_acc": history[-1]["val_acc"],
        "total_steps": global_step,
        "elapsed_sec": elapsed,
        "history": history,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Leave-one-domain-out MAML and convergence speed evaluation")
    p.add_argument("--backbone", default="dinov3")
    p.add_argument("--domains", nargs="+", default=DOMAINS)
    p.add_argument("--n-sources", type=int, default=3, help="Number of other domains to sample for MAML prep")
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--meta-lr", type=float, default=1e-6)
    p.add_argument("--meta-episodes", type=int, default=300)
    p.add_argument("--meta-batch", type=int, default=1)
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--adapt-lr", type=float, default=1e-4)
    p.add_argument("--adapt-epochs", type=int, default=3)
    p.add_argument("--adapt-batch-size", type=int, default=32)
    p.add_argument("--eval-every-steps", type=int, default=10)
    p.add_argument("--target-acc", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-prefix", default="maml_convergence")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required to run MAML training and evaluation.")

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)

    summary = {}
    print(f"Beginning Cross-Domain MAML Convergence Run across domains: {args.domains}\n")

    for target_domain in args.domains:
        available_sources = [d for d in args.domains if d != target_domain]
        # Select source domains for MAML meta-training
        rng = random.Random(args.seed + hash(target_domain))
        if args.n_sources and args.n_sources < len(available_sources):
            selected_sources = rng.sample(available_sources, args.n_sources)
        else:
            selected_sources = available_sources

        print(f"==================================================")
        print(f"Target Domain: {target_domain}")
        print(f"Source Domains for MAML: {selected_sources}")
        print(f"==================================================")

        # 1. Initialize fresh backbone
        model = backbone(args.backbone, num_classes=args.n_way).to(device)

        # 2. Meta-train on the source domains
        tasks = make_maml_tasks(selected_sources, args, device)
        meta_train_maml(model, tasks, args, device)

        # Save checkpoint for this domain split
        ckpt_path = Path("models") / f"{args.output_prefix}_target_{target_domain}.pt"
        torch.save(model.state_dict(), ckpt_path)

        # 3. Evaluate speed of convergence on the target domain
        convergence_metrics = evaluate_target_convergence(model, target_domain, args, device)
        summary[target_domain] = {
            "source_domains": selected_sources,
            "step_to_target": convergence_metrics["step_to_target"],
            "final_acc": convergence_metrics["final_acc"],
            "total_steps": convergence_metrics["total_steps"],
            "elapsed_sec": convergence_metrics["elapsed_sec"],
            "history": convergence_metrics["history"],
        }

        print(
            f"Finished {target_domain} | Steps to {args.target_acc:.0%}: {convergence_metrics['step_to_target']} | "
            f"Final Acc: {convergence_metrics['final_acc']:.4f}\n"
        )

    # Save summary of all domains
    out_file = Path("results") / f"{args.output_prefix}_all.json"
    out_file.write_text(json.dumps({"args": vars(args), "results": summary}, indent=2))
    print(f"Done! Consolidated results saved to {out_file}")


if __name__ == "__main__":
    main()
