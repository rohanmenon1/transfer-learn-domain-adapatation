import argparse
from collections import defaultdict
import json
import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common import DomainNet, backbone


class RemappedDataset(Dataset):
    """Top-level dataset class to avoid multiprocessing pickling issues."""
    def __init__(self, dataset, selected_indices, label_map):
        self.dataset = dataset
        self.selected_indices = selected_indices
        self.label_map = label_map

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, idx):
        x, original_label = self.dataset[self.selected_indices[idx]]
        return x, self.label_map[original_label]


@torch.no_grad()
def evaluate(model, loader, device):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="maml_domainnet")
    parser.add_argument("--domain", default="sketch")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every-steps", type=int, default=10)
    parser.add_argument("--target-acc", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MAML config
    with open(f"results/{args.checkpoint}.json") as f:
        config = json.load(f)["args"]

    n_way = config["n_way"]

    # Load architecture and weights
    model = backbone(config["backbone"], num_classes=n_way)
    model.load_state_dict(
        torch.load(f"models/{args.checkpoint}.pt", map_location=device, weights_only=True)
    )
    model.to(device)

    # Load target domain splits
    train_dataset = DomainNet(args.domain, "train", train=True, per_class=config.get("per_class", 50), seed=args.seed)
    test_dataset = DomainNet(args.domain, "test", train=False, seed=args.seed)

    # Select classes present in both splits
    by_class = defaultdict(list)
    for i, (_, label) in enumerate(train_dataset.items):
        by_class[label].append(i)

    eligible = [label for label, ids in by_class.items() if len(ids) >= args.batch_size]
    if len(eligible) < n_way:
        raise ValueError(f"Only {len(eligible)} classes have >= {args.batch_size} samples, but n_way={n_way}.")

    selected = sorted(random.sample(eligible, n_way))
    label_map = {label: i for i, label in enumerate(selected)}

    train_indices = [i for i, (_, label) in enumerate(train_dataset.items) if label in label_map]
    test_indices = [i for i, (_, label) in enumerate(test_dataset.items) if label in label_map]

    train_loader = DataLoader(
        RemappedDataset(train_dataset, train_indices, label_map),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
    )
    test_loader = DataLoader(
        RemappedDataset(test_dataset, test_indices, label_map),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Initial zero-shot evaluation (step 0)
    init_loss, init_acc = evaluate(model, test_loader, device)
    print(f"Step 0 (Pre-adaptation) | Test Loss: {init_loss:.4f} | Test Acc: {init_acc:.4f}")

    global_step = 0
    step_to_target = None
    history = [{"step": 0, "test_loss": init_loss, "test_acc": init_acc}]

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            global_step += 1

            # Frequent step-level evaluation to capture convergence trajectory
            if global_step % args.eval_every_steps == 0:
                val_loss, val_acc = evaluate(model, test_loader, device)
                history.append({"step": global_step, "test_loss": val_loss, "test_acc": val_acc})
                print(f"Epoch {epoch+1} | Step {global_step} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                if step_to_target is None and val_acc >= args.target_acc:
                    step_to_target = global_step
                    print(f"--> Reached target accuracy ({args.target_acc:.0%}) at step {global_step}!")

                model.train()

    elapsed = time.time() - start_time
    print(f"\n--- Convergence Summary ---")
    print(f"Total steps: {global_step} in {elapsed:.2f}s")
    print(f"Steps to {args.target_acc:.0%} accuracy: {step_to_target if step_to_target else 'Not reached'}")

    # Save metrics for plotting convergence curves
    log_file = f"results/{args.checkpoint}_convergence_{args.domain}.json"
    with open(log_file, "w") as f:
        json.dump({"history": history, "steps_to_target": step_to_target, "elapsed_sec": elapsed}, f, indent=2)
    print(f"Convergence history saved to {log_file}")


if __name__ == "__main__":
    main()
