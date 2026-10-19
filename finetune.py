import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import DOMAINS, NUM_CLASSES, DomainNet, accuracy, backbone, test_loader

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="dinov3")
p.add_argument("--source", default="real")
p.add_argument("--targets", nargs="+", default=DOMAINS)
p.add_argument("--per-class", type=int, default=50)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--lr", type=float, default=3e-5)
p.add_argument("--head-lr", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed)
model = backbone(args.backbone, NUM_CLASSES)
train = DataLoader(DomainNet(args.source, "train", train=True, per_class=args.per_class, seed=args.seed),
                   batch_size=64, shuffle=True, drop_last=True, num_workers=10, persistent_workers=True)


head = list(model.get_classifier().parameters())
body = [p for p in model.parameters() if all(p is not h for h in head)]
opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.head_lr}], weight_decay=0.05)
steps, warmup = args.epochs * len(train), len(train)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / steps))))

out = Path(f"results/ft_{args.backbone}_{args.source}_pc{args.per_class}_s{args.seed}.json")
out.parent.mkdir(exist_ok=True)


def save(**extra):
    out.write_text(json.dumps({"args": vars(args), "history": history, **extra}, indent=2))


subsets = {d: test_loader(d, 2000, args.seed) for d in args.targets}
history = []
for epoch in range(args.epochs + 1):
    if epoch:
        for x, y in train:
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(x.cuda()).float(), y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    history.append({"step": epoch * len(train), **{d: accuracy(model, l) for d, l in subsets.items()}})
    print(history[-1], flush=True)
    save()

Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), f"models/{out.stem}.pt")
