import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import NUM_CLASSES, DomainNet, backbone

p = argparse.ArgumentParser()
p.add_argument("--init", default="none")
p.add_argument("--target", default="clipart")
p.add_argument("--per-class", type=int, default=50)
p.add_argument("--max-epochs", type=int, default=10)
p.add_argument("--patience", type=float, default=2)
p.add_argument("--eval-every", type=int, default=100)
args = p.parse_args()

out = Path(f"results/converge_{args.init}_to-{args.target}.json")
if out.exists():
    raise SystemExit
torch.manual_seed(0)
test = DataLoader(DomainNet(args.target, "test", limit=2000), batch_size=256, num_workers=10)
X, Y = zip(*[(x.cuda().bfloat16(), y.cuda()) for x, y in test])


@torch.no_grad()
def accuracy(model):
    model.eval()
    with torch.autocast("cuda", torch.bfloat16):
        hits = sum((model(x).argmax(1) == y).sum().item() for x, y in zip(X, Y))
    model.train()
    return 100 * hits / len(test.dataset)


model = backbone("dinov3", NUM_CLASSES)
data = DomainNet(args.target, "train", train=True, per_class=args.per_class)
if args.init != "none":
    model.load_state_dict(torch.load(f"models/{args.init}.pt"))
train = DataLoader(data, batch_size=64, shuffle=True, drop_last=True, num_workers=10, persistent_workers=True)
head = list(model.get_classifier().parameters())
body = [q for q in model.parameters() if all(q is not h for h in head)]
opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / 100, 1))
window = int(args.patience * len(train))
curve = [[0, accuracy(model)]]
step, done = 0, False
while not done:
    for x, y in train:
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(x.cuda()).float(), y.cuda())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % args.eval_every == 0:
            curve.append([step, accuracy(model)])
            print(curve[-1], flush=True)
            old = [a for s, a in curve if s <= step - window]
            done = step >= args.max_epochs * len(train) or bool(old) and max(a for s, a in curve if s > step - window) < max(old) + 0.5
            if done:
                break

plateau = max(a for _, a in curve)
out.write_text(json.dumps({"args": vars(args), "steps_per_epoch": len(train), "plateau": plateau,
                           "converged_at": next(s for s, a in curve if a >= 0.95 * plateau), "curve": curve}, indent=2))
