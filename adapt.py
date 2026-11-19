import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import NUM_CLASSES, TEST_TF, TRAIN_TF, DomainNet, backbone, linear_probe

p = argparse.ArgumentParser()
p.add_argument("--init", default="none")
p.add_argument("--target", default="clipart")
p.add_argument("--k", type=int, default=10)
p.add_argument("--steps", type=int, default=500)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = p.parse_args()

out = Path(f"results/adapt_{args.init}_to-{args.target}_k{args.k}.json")
runs = json.loads(out.read_text())["runs"] if out.exists() else {}
evals = sorted({0, *range(10, 101, 10), *range(150, args.steps + 1, 50)})

test = DataLoader(DomainNet(args.target, "test", per_class=20, seed=7), batch_size=256, num_workers=10)
X, Y = zip(*[(x.cuda().bfloat16(), y.cuda()) for x, y in test])


@torch.no_grad()
def accuracy(model):
    model.eval()
    with torch.autocast("cuda", torch.bfloat16):
        hits = sum((model(x).argmax(1) == y).sum().item() for x, y in zip(X, Y))
    model.train()
    return 100 * hits / len(test.dataset)


for seed in args.seeds:
    if str(seed) in runs:
        continue
    torch.manual_seed(seed)
    model = backbone("dinov3", NUM_CLASSES)
    shots = DomainNet(args.target, "train", per_class=args.k, seed=1000 + seed)
    if args.init == "none":
        shots.tf = TEST_TF
        linear_probe(model, shots)
        shots.tf = TRAIN_TF
    else:
        model.load_state_dict(torch.load(f"models/{args.init}.pt"))
    train = DataLoader(shots, batch_size=64, shuffle=True, drop_last=True, num_workers=10, persistent_workers=True)
    head = list(model.get_classifier().parameters())
    body = [q for q in model.parameters() if all(q is not h for h in head)]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    warmup = len(train)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * s / args.steps))))
    curve = [[0, accuracy(model)]]
    step = 0
    while step < args.steps:
        for x, y in train:
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(x.cuda()).float(), y.cuda())
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step in evals:
                curve.append([step, accuracy(model)])
                print(seed, curve[-1], flush=True)
            if step == args.steps:
                break
    runs[str(seed)] = curve
    out.write_text(json.dumps({"args": vars(args), "runs": runs}, indent=2))
