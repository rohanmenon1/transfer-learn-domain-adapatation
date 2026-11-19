import argparse
import json
import math
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from common import DOMAINS, NORMALIZE, NUM_CLASSES, DomainNet

p = argparse.ArgumentParser()
p.add_argument("--arch", default="resnet18")
p.add_argument("--source", default="clipart")
p.add_argument("--pretrained", action="store_true")
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--size", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
args = p.parse_args()

torch.manual_seed(0)
model = timm.create_model(f"{args.arch}.tv_in1k", pretrained=args.pretrained, num_classes=NUM_CLASSES).cuda().to(memory_format=torch.channels_last)
data = DomainNet(args.source, "train")
data.tf = T.Compose([T.RandomResizedCrop(args.size, (0.5, 1)), T.RandomHorizontalFlip(), T.ToTensor(), NORMALIZE])
train = DataLoader(data, batch_size=128, shuffle=True, drop_last=True, num_workers=10, persistent_workers=True)
test_tf = T.Compose([T.Resize(args.size * 8 // 7), T.CenterCrop(args.size), T.ToTensor(), NORMALIZE])


def loader(domain, limit=0):
    d = DomainNet(domain, "test", limit=limit)
    d.tf = test_tf
    return DataLoader(d, batch_size=256, num_workers=10)


@torch.no_grad()
def accuracy(l):
    model.eval()
    hits = 0
    for x, y in l:
        with torch.autocast("cuda", torch.bfloat16):
            hits += (model(x.cuda().to(memory_format=torch.channels_last)).argmax(1).cpu() == y).sum().item()
    model.train()
    return hits / len(l.dataset)


opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
steps, warmup = args.epochs * len(train), len(train)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / steps))))
name = f"{'pre' if args.pretrained else 'scratch'}_{args.arch}_{args.source}"
out = Path(f"results/{name}.json")
out.parent.mkdir(exist_ok=True)
subsets = {d: loader(d, 2000) for d in DOMAINS}
Path("models").mkdir(exist_ok=True)
last = Path(f"models/{out.stem}_last.pt")
history = []
if last.exists():
    ckpt = torch.load(last)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    history = ckpt["history"]
for epoch in range(len(history), args.epochs + 1):
    if epoch:
        for x, y in train:
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(x.cuda().to(memory_format=torch.channels_last)).float(), y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    history.append({"epoch": epoch, **{d: accuracy(l) for d, l in subsets.items()}})
    print(history[-1], flush=True)
    out.write_text(json.dumps({"args": vars(args), "history": history}, indent=2))
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "history": history}, last)

torch.save(model.state_dict(), f"models/{name}.pt")
final = {d: accuracy(loader(d)) for d in DOMAINS}
print("final", final)
out.write_text(json.dumps({"args": vars(args), "history": history, "final": final}, indent=2))
last.unlink()
