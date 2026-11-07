"""Adaptation speed: fine-tune an init on k labelled images/class of a target domain and track accuracy.

  python kd_adapt.py --init models/expert_dinov3_real_s0.pt --target sketch
  python kd_adapt.py --init dinov3 --target sketch --lpft    # pretrained DINOv3 only (no DomainNet training)

Protocol: k = 10 labelled images per class of the target (train split), 500 steps, batch 64, the finetune.py
optimiser (AdamW, backbone lr 3e-5, head lr 1e-3, warm-up + cosine). Accuracy is measured on a fixed,
class-balanced subset of the target test set (20 images/class), every 10 steps up to 100, then every 50.
Reported: steps to reach 80, 90 and 95% of the in-domain ceiling (kd_common.CEILING), and the area under the
accuracy curve (AUC = mean accuracy over the 500 steps). Each seed draws a different k-shot sample.

Accepts any timm DINOv3 ViT-S checkpoint (state dict, or nested under 'model'/'state_dict'); a head with
another shape is replaced by a fresh 345-way head.
Output: results/adapt_<name>_to-<target>_k<k>_<tag>.json (resumable per seed).
"""
import argparse
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import kd_common as K
from kd_losses import LOSSES


def eval_steps(total):
    return sorted({0, *range(10, min(total, 100) + 1, 10), *range(150, total + 1, 50), total})


def shots(target, k, seed):
    by = defaultdict(list)
    for it in K.DomainNet(target, "train", per_class=K.PER_CLASS, seed=K.SAMPLE_SEED).items:
        by[it[1]].append(it)
    r = random.Random(1000 + seed)
    out = [x for c in sorted(by) for x in r.sample(by[c], min(k, len(by[c])))]
    return out[:: max(1, len(out) // 64)] if K.SMOKE else out


def eval_set(target, per_class):
    """Fixed class-balanced subset of the target TEST set (same for every init and seed), kept on the GPU."""
    by = defaultdict(list)
    for it in K.DomainNet(target, "test").items:
        by[it[1]].append(it)
    r = random.Random(7)
    items = [x for c in sorted(by) for x in r.sample(by[c], min(per_class, len(by[c])))]
    if K.SMOKE:
        items = items[:128]
    xs, ys = [], []
    for x, y in K.test_loader(items):
        xs.append(x.cuda())
        ys.append(y.cuda())
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def acc_on(model, X, Y):
    model.eval()
    hits = 0
    for j in range(0, len(X), K.EVAL_BATCH):
        with torch.autocast("cuda", torch.bfloat16):
            hits += (model(K.norm(X[j:j + K.EVAL_BATCH])).argmax(1) == Y[j:j + K.EVAL_BATCH]).sum().item()
    model.train()
    return round(100.0 * hits / len(Y), 2)


def summarize(curve, thr, total, target=None):
    steps = [s for s, _ in curve]
    accs = [a for _, a in curve]
    hit = next((s for s, a in curve if a >= thr), None)
    multi = {str(f): next((s for s, a in curve if a >= K.threshold(target, f)), None) for f in K.THRESHOLDS} if target else {}
    auc = float(np.trapezoid(accs, steps) / (steps[-1] - steps[0])) if len(steps) > 1 else accs[0]
    return {"steps_to_thr": hit, "censored": hit is None, "steps_to": multi, "auc": round(auc, 2), "start": accs[0], "end": accs[-1]}


@torch.no_grad()
def _features(model, items):
    model.eval()
    fs, ys = [], []
    for x, y in K.test_loader(items):
        with torch.autocast("cuda", torch.bfloat16):
            fs.append(model.forward_head(model.forward_features(K.norm(x.cuda())), pre_logits=True).float())
        ys.append(y.cuda())
    model.train()
    return torch.cat(fs), torch.cat(ys)


def linear_probe_head(model, items, l2=1e-4):
    """LP step of LP-FT: multinomial logistic regression (L-BFGS) on frozen features of the shots
    (centre crop, no augmentation); the result becomes the model's head."""
    X, Y = _features(model, items)
    head = model.get_classifier()
    W = torch.zeros_like(head.weight, dtype=torch.float32, requires_grad=True)
    b = torch.zeros_like(head.bias, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W.t() + b, Y) + l2 * (W ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        head.weight.copy_(W)
        head.bias.copy_(b)
        return round(100.0 * ((X @ W.t() + b).argmax(1) == Y).float().mean().item(), 2)


def run_seed(a, seed, X, Y):
    torch.manual_seed(seed)
    model = K.new_model()
    rep = K.load_init(model, a.init) if a.init != "dinov3" else {"loaded": "dinov3 pretrained only"}
    sh = shots(a.target, a.k, seed)
    if a.lpft:
        rep["lpft_train_acc"] = linear_probe_head(model, sh)
    total = K.SMOKE_STEPS if K.SMOKE else a.steps
    per_ep = max(1, len(sh) // K.BATCH)
    dl = K.view_loader(sh, np.arange(len(sh)), total // per_ep + 2, seed=seed, view_seed=100 + seed)
    opt, sched = K.make_opt(model, total, per_ep)
    when = set(eval_steps(total))
    curve = [(0, acc_on(model, X, Y))]
    step, t0 = 0, time.time()
    for x, y, _, _ in dl:
        x, y = K.norm(x.cuda(non_blocking=True)), y.cuda(non_blocking=True)
        with torch.autocast("cuda", torch.bfloat16):
            zs = model(x)
        loss, _ = LOSSES["erm"](zs.float(), None, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
        if step in when:
            curve.append((step, acc_on(model, X, Y)))
            K.log(f"seed {seed} step {step}/{total} acc {curve[-1][1]:.2f}")
        if step >= total:
            break
    return {"seed": seed, "curve": curve, "init_load": rep, "n_shots": len(sh), "sec": round(time.time() - t0),
            **summarize(curve, K.threshold(a.target), total, a.target)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="checkpoint path, or 'dinov3' for the pretrained backbone only")
    p.add_argument("--name", help="label used in the output file name (default: checkpoint stem)")
    p.add_argument("--target", required=True, choices=K.DOMAINS)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--eval-per-class", type=int, default=20)
    p.add_argument("--lpft", action="store_true", help="fit the linear head on the shots first (LP-FT, Kumar et al. 2022)")
    p.add_argument("--tag", default="v2", help="suffix of the output file name")
    a = p.parse_args()
    name = a.name or (Path(a.init).stem if a.init != "dinov3" else "dinov3")
    out = K.RESULTS / f"adapt_{name}_to-{a.target}_k{a.k}{'_' + a.tag if a.tag else ''}.json"
    res = K.load_json(out, {}) or {}
    if res.get("complete"):
        return
    runs = {r["seed"]: r for r in res.get("runs", [])}
    X, Y = eval_set(a.target, a.eval_per_class)
    thr = K.threshold(a.target)
    for s in a.seeds:
        if s in runs:
            continue
        runs[s] = run_seed(a, s, X, Y)
        K.log(f"seed {s}: steps to {thr:.1f}% = {runs[s]['steps_to_thr']} | AUC {runs[s]['auc']}")
        res = {"kind": "adapt", "init": str(a.init), "name": name, "target": a.target, "k": a.k,
               "threshold": round(thr, 2), "threshold_def": f"{K.THRESHOLD_FRAC} x ceiling {K.CEILING[a.target]}",
               "eval_images": len(Y), "steps": a.steps, "lpft": a.lpft, "thresholds": list(K.THRESHOLDS), "runs": [runs[k] for k in sorted(runs)], "complete": False}
        K.save_json(out, res)
    R = res["runs"]
    cs = np.array([[acc for _, acc in r["curve"]] for r in R])
    hits = [r["steps_to_thr"] for r in R]
    res.update({
        "mean_curve": [[s, round(float(v), 2)] for (s, _), v in zip(R[0]["curve"], cs.mean(0))],
        "auc_mean": round(float(np.mean([r["auc"] for r in R])), 2),
        "auc_sd": round(float(np.std([r["auc"] for r in R], ddof=1)), 2) if len(R) > 1 else 0.0,
        "start_mean": round(float(np.mean([r["start"] for r in R])), 2),
        "steps_to_thr_median": (float(np.median([h if h is not None else np.inf for h in hits]))
                                if sum(h is not None for h in hits) * 2 > len(hits) else None),
        "n_reached": sum(h is not None for h in hits),
        "steps_to_median": {f: (float(np.median([r["steps_to"][f] if r["steps_to"][f] is not None else np.inf for r in R]))
                                if sum(r["steps_to"][f] is not None for r in R) * 2 > len(R) else None)
                            for f in R[0].get("steps_to", {})},
        "complete": True})
    K.save_json(out, res)


if __name__ == "__main__":
    main()
