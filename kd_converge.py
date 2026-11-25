"""Phase 2 convergence test: the team's shared protocol (converge.py on the fine-tuning branch), for KD inits.

  python kd_converge.py --init models/kd-vitb_dinov3_src-clipart+infograph_T4_w0.5_s0.pt --target painting
  python kd_converge.py --init none --target painting      # pretrained DINOv3, fresh 345-way head

Same protocol as converge.py, line for line:
  - target training data: 50 images/class from the target train split (seeded per-class sample, seed 0)
  - test set: 2,000 random images from the target test split (seed 0), fixed for every init
  - AdamW (backbone lr 3e-5, head lr 1e-3, weight decay 0.05), batch 64, 224 px,
    linear warm-up over 100 steps, then a constant learning rate
  - accuracy every 100 steps; stop when the best accuracy of the last 2 epochs is not at least
    0.5 points above the best accuracy before them, or after 10 epochs
  - converged_at = first evaluated step whose accuracy reaches 95% of the best accuracy (plateau)
Only the data loading differs: images are read from DOMAINNET_ROOT (official *_train.txt / *_test.txt lists)
through kd_common, with the same sampling code, and the entry point is guarded so it runs on Windows.

Output: results/converge_<name>_to-<target>.json (same fields as converge.py, plus a few extras).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import kd_common as K

WARMUP_STEPS = 100
CONVERGE_FRAC = 0.95
MIN_GAIN = 0.5


@torch.no_grad()
def accuracy(model, X, Y):
    model.eval()
    hits = 0
    for j in range(0, len(X), K.EVAL_BATCH):
        with torch.autocast("cuda", torch.bfloat16):
            hits += (model(K.norm(X[j:j + K.EVAL_BATCH])).argmax(1) == Y[j:j + K.EVAL_BATCH]).sum().item()
    model.train()
    return 100 * hits / len(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", default="none", help="checkpoint path, or 'none' for the pretrained backbone only")
    p.add_argument("--name", help="label used in the output file name (default: checkpoint stem)")
    p.add_argument("--target", default="clipart", choices=K.DOMAINS)
    p.add_argument("--per-class", type=int, default=50)
    p.add_argument("--max-epochs", type=int, default=10)
    p.add_argument("--patience", type=float, default=2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    name = a.name or (Path(a.init).stem if a.init != "none" else "none")
    out = K.RESULTS / f"converge_{name}_to-{a.target}{'' if a.seed == 0 else f'_s{a.seed}'}.json"
    if out.exists():
        K.log(f"exists, skipping: {out.name}")
        return

    torch.manual_seed(a.seed)
    test = K.test_items(a.target, 2000, 0)
    xs, ys = [], []
    for x, y in K.test_loader(test):
        xs.append(x.cuda())
        ys.append(y.cuda())
    X, Y = torch.cat(xs), torch.cat(ys)

    model = K.new_model()
    rep = K.load_init(model, a.init) if a.init != "none" else {"loaded": "dinov3 pretrained only"}
    items = K.DomainNet(a.target, "train", per_class=a.per_class, seed=0).items
    eval_every = a.eval_every
    if K.SMOKE:
        items, eval_every = items[::20], 5
    per_ep = len(items) // K.BATCH
    max_steps = a.max_epochs * per_ep
    train = K.view_loader(items, np.arange(len(items)), a.max_epochs + 1, seed=a.seed, view_seed=a.seed)

    head = list(model.get_classifier().parameters())
    hid = {id(q) for q in head}
    body = [q for q in model.parameters() if id(q) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": 3e-5}, {"params": head, "lr": 1e-3}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / WARMUP_STEPS, 1))
    window = int(a.patience * per_ep)

    curve = [[0, accuracy(model, X, Y)]]
    K.log(f"{name} -> {a.target}: {len(items)} train images, {per_ep} steps/epoch, start acc {curve[0][1]:.2f}")
    step, done, t0 = 0, False, time.time()
    model.train()
    for x, y, _, _ in train:
        with torch.autocast("cuda", torch.bfloat16):
            loss = F.cross_entropy(model(K.norm(x.cuda(non_blocking=True))).float(), y.cuda(non_blocking=True))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step % eval_every == 0:
            curve.append([step, accuracy(model, X, Y)])
            K.log(f"step {step} acc {curve[-1][1]:.2f} | {(time.time() - t0) / 60:.1f} min")
            old = [acc for s, acc in curve if s <= step - window]
            done = step >= max_steps or bool(old) and max(acc for s, acc in curve if s > step - window) < max(old) + MIN_GAIN
        if done:
            break
    if not done:  # loader ran out before a stop (only possible in smoke tests)
        curve.append([step, accuracy(model, X, Y)])

    plateau = max(acc for _, acc in curve)
    K.save_json(out, {
        "args": vars(a), "name": name, "steps_per_epoch": per_ep, "plateau": plateau,
        "converged_at": next(s for s, acc in curve if acc >= CONVERGE_FRAC * plateau), "curve": curve,
        "protocol": "converge.py (fine-tuning branch): 50/class target train, 2000 test images, warm-up 100 then "
                    "constant lr, eval every 100, stop when last 2 epochs gain < 0.5 pts or 10 epochs, "
                    "converged_at = first step >= 95% of plateau",
        "n_train": len(items), "n_test": len(Y), "stopped_at": step, "init_load": rep,
        "sec": round(time.time() - t0)})
    K.log(f"done: plateau {plateau:.2f}, converged at step {next(s for s, acc in curve if acc >= CONVERGE_FRAC * plateau)}, "
          f"stopped at {step}")


if __name__ == "__main__":
    main()
