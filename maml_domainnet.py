
# maml_domainnet.py
import sys 

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import higher
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from common import DOMAINS, DomainNet, backbone
from torch.nn.attention import sdpa_kernel, SDPBackend


def parse_args():
    p = argparse.ArgumentParser(description="MAML on DomainNet")
    #TODO: change backbone to resnet NVM
    p.add_argument("--backbone", default="dinov3")
    p.add_argument("--domains", nargs="+", default=["painting"])
    p.add_argument("--n-way", type=int, default=5)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--q-query", type=int, default=5)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--meta-lr", type=float, default=1e-6)
    p.add_argument("--episodes", type=int, default=550)
    p.add_argument("--meta-batch", type=int, default=1)
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default="maml_domainnet")
    return p.parse_args()


class EpisodeSampler:
    """Samples N-way K-shot classification episodes."""

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
            c for c, indices in self.by_class.items()
            if len(indices) >= needed
        ]

        if len(self.classes) < n_way:
            raise ValueError(
                f"Need {n_way} classes with at least {needed} images "
                f"each; found {len(self.classes)}."
            )

    def sample(self):
        # Select classes and remap their labels to [0, n_way).
        classes = self.rng.sample(self.classes, self.n_way)
        support_x, support_y = [], []
        query_x, query_y = [], []

        for new_label, original_label in enumerate(classes):
            indices = self.rng.sample(
                self.by_class[original_label],
                self.k_shot + self.q_query,
            )

            support_ids = indices[:self.k_shot]
            query_ids = indices[self.k_shot:]

            for i in support_ids:
                x, _ = self.dataset[i]
                support_x.append(x)
                support_y.append(new_label)

            for i in query_ids:
                x, _ = self.dataset[i]
                query_x.append(x)
                query_y.append(new_label)

        device = torch.device("cuda")
        return (
            torch.stack(support_x).to(device),
            torch.tensor(support_y, device=device),
            torch.stack(query_x).to(device),
            torch.tensor(query_y, device=device),
        )


def make_tasks(args):
    """Create one sampler for each source domain."""
    tasks = {}

    for i, domain in enumerate(args.domains):
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
        print(
            f"{domain}: {len(dataset):,} images, "
            f"{len(tasks[domain].classes)} eligible classes"
        )

    return tasks


def sample_task(tasks, rng):
    domain = rng.choice(list(tasks.keys()))
    return domain, tasks[domain].sample()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA GPU.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output_dir = Path("results")
    model_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # The classifier outputs N-way episode-specific labels.
    model = backbone(args.backbone, num_classes=args.n_way)
    model.train()

    tasks = make_tasks(args)
    rng = random.Random(args.seed + 1000)

    # Outer-loop optimizer updates the MAML initialization.
    meta_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.meta_lr,
        weight_decay=0.01,
    )

    history = []

    for episode in range(1, args.episodes + 1):
        meta_optimizer.zero_grad(set_to_none=True)
        meta_loss_value = 0.0
        meta_acc_value = 0.0

        for _ in range(args.meta_batch):
            domain, batch = sample_task(tasks, rng)
            support_x, support_y, query_x, query_y = batch

            # The inner optimizer is differentiable through higher.
            inner_optimizer = torch.optim.SGD(
                model.parameters(),
                lr=args.inner_lr,
            )

            with sdpa_kernel(SDPBackend.MATH):
                with higher.innerloop_ctx(
                    model,
                    inner_optimizer,
                    copy_initial_weights=False,
                    track_higher_grads=True,
                ) as (fast_model, diff_optimizer):

                    # Inner loop: adapt using the support set.
                    for _ in range(args.inner_steps):
                        support_logits = fast_model(support_x)
                        support_loss = F.cross_entropy(
                            support_logits.float(), support_y
                        )
                        diff_optimizer.step(support_loss)

                    # Outer objective: performance after adaptation.
                    query_logits = fast_model(query_x)
                    query_loss = F.cross_entropy(
                        query_logits.float(), query_y
                    )

                    query_acc = (
                        query_logits.argmax(dim=1) == query_y
                    ).float().mean()

                    # Backpropagate through the adaptation process.
                    (query_loss / args.meta_batch).backward()

                    meta_loss_value += query_loss.item() / args.meta_batch
                    meta_acc_value += query_acc.item() / args.meta_batch

        # Outer loop: improve the initialization across tasks.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        meta_optimizer.step()

        record = {
            "episode": episode,
            "meta_loss": meta_loss_value,
            "query_accuracy": meta_acc_value,
        }
        history.append(record)

        if episode == 1 or episode % 10 == 0:
            print(
                f"Episode {episode:5d}/{args.episodes} | "
                f"query loss {meta_loss_value:.4f} | "
                f"query acc {meta_acc_value:.4f}",
                flush=True,
            )

        if episode % 100 == 0 or episode == args.episodes:
            torch.save(
                model.state_dict(),
                model_dir / f"{args.output}.pt",
            )
            (output_dir / f"{args.output}.json").write_text(
                json.dumps(
                    {
                        "args": vars(args),
                        "history": history,
                    },
                    indent=2,
                )
            )

    print(f"Saved model to models/{args.output}.pt")
    print(f"Saved metrics to results/{args.output}.json")


if __name__ == "__main__":
    main()
