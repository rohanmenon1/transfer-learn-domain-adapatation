import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import DOMAINS, NUM_CLASSES, DomainNet, backbone

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="dinov3")
p.add_argument("--domains", nargs="+", default=DOMAINS)
p.add_argument("--per-class", type=int, default=50)
p.add_argument("--scratch", action="store_true")
args = p.parse_args()
name = args.backbone + ("_scratch" if args.scratch else "")

torch.manual_seed(0)
model = backbone(args.backbone, pretrained=not args.scratch).eval()


@torch.no_grad()
def features(domain, split, per_class=0):
    X, y = [], []
    for x, t in DataLoader(DomainNet(domain, split, per_class=per_class), batch_size=256, num_workers=10):
        with torch.autocast("cuda", torch.bfloat16):
            X.append(model(x.cuda()).float())
        y.append(t.cuda())
    return torch.cat(X), torch.cat(y)


def fit(X, y, epochs=100):
    torch.manual_seed(0)
    head = torch.nn.Linear(X.shape[1], NUM_CLASSES).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    batches = [i for _ in range(epochs) for i in torch.randperm(len(X), device="cuda").split(1024)]
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, len(batches))
    for i in batches:
        opt.zero_grad()
        F.cross_entropy(head(X[i]), y[i]).backward()
        opt.step()
        sched.step()
    return head


test = {d: features(d, "test") for d in args.domains}
acc = {}
for src in args.domains:
    X, y = features(src, "train", args.per_class)
    mean, std = X.mean(0), X.std(0)
    head = fit((X - mean) / std, y)
    for tgt, (Xt, yt) in test.items():
        acc[f"{src}->{tgt}"] = (head((Xt - mean) / std).argmax(1) == yt).float().mean().item()
        print(f"{src} -> {tgt}: {acc[f'{src}->{tgt}']:.4f}")

Path("results").mkdir(exist_ok=True)
Path(f"results/probe_{name}_pc{args.per_class}.json").write_text(json.dumps(acc, indent=2))
