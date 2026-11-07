"""Domain distances on frozen DINOv3 ViT-S features (pretrained, no DomainNet training).

Fréchet distance (FID-style) between Gaussian fits of the feature distributions:
  FD = |mu1 - mu2|^2 + Tr(C1 + C2 - 2 (C1 C2)^1/2)
- pairwise: single domain -> single domain (6 x 6)
- lodo: pooled 5 sources (mixture of their Gaussians, weighted by image count) -> held-out target
Features come from 20 imgs/class of each domain's seeded TRAIN sample (no test images), Resize 256 -> CenterCrop 224.
OTDD (Alvarez-Melis & Fusi 2020) with the Gaussian approximation for the label-to-label cost:
  cost((x,y),(x',y')) = |x - x'|^2 + W2^2(N_y, N_y'),  N_c = diagonal Gaussian of class c's features;
  exact OT (POT emd) between 2,000-point subsamples; OTDD = sqrt(OT cost).
Output: results/distance_dinov3.json (keys: frechet / otdd -> pairwise, lodo).
"""
import random
import time
from collections import defaultdict

import numpy as np
import torch
from scipy import linalg

import kd_common as K
from kd_common import DOMAINS

PER_CLASS = 20


N_OT = 2000


def otdd(fa, ya, fb, yb, seed=0):
    import ot
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(fa), min(N_OT, len(fa)), replace=False)
    ib = rng.choice(len(fb), min(N_OT, len(fb)), replace=False)

    def cls_stats(f, y):
        st = np.zeros((345, 2 * f.shape[1]))
        for c in range(345):
            g = f[y == c]
            if len(g):
                st[c] = np.concatenate([g.mean(0), g.std(0)])
        return st
    sa, sb = cls_stats(fa, ya), cls_stats(fb, yb)
    lab = sqdist(sa, sb)                                             # W2^2 between diagonal Gaussians
    M = sqdist(fa[ia], fb[ib]) + lab[ya[ia]][:, yb[ib]]
    w = np.full(len(ia), 1 / len(ia)), np.full(len(ib), 1 / len(ib))
    return float(np.sqrt(ot.emd2(*w, M, numItermax=2_000_000)))


def sqdist(a, b):
    return np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)


def frechet(m1, c1, m2, c2):
    s = linalg.sqrtm(c1 @ c2)
    s = s[0] if isinstance(s, tuple) else s
    s = s.real
    return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * s))


@torch.no_grad()
def main():
    t0 = time.time()
    model = K.timm.create_model(K.BACKBONES[K.STUDENT], pretrained=True, num_classes=0).cuda().eval()
    stats, feats = {}, {}
    for d in DOMAINS:
        by = defaultdict(list)
        for it in K.DomainNet(d, "train", per_class=K.PER_CLASS, seed=K.SAMPLE_SEED).items:
            by[it[1]].append(it)
        r = random.Random(5)
        items = [x for c in sorted(by) for x in r.sample(by[c], min(PER_CLASS, len(by[c])))]
        fs = []
        for x, _ in K.test_loader(items):
            with torch.autocast("cuda", torch.bfloat16):
                fs.append(model(K.norm(x.cuda(non_blocking=True))).float().cpu())
        f = torch.cat(fs).double().numpy()
        stats[d] = (f.mean(0), np.cov(f, rowvar=False), len(f))
        feats[d] = (f, np.array([y for _, y in items]))
        K.log(f"features {d}: {f.shape}")
    pair = {a: {b: round(frechet(*stats[a][:2], *stats[b][:2]), 2) for b in DOMAINS} for a in DOMAINS}
    lodo = {}
    for t in DOMAINS:
        src = [d for d in DOMAINS if d != t]
        w = np.array([stats[d][2] for d in src], float)
        w /= w.sum()
        mu = sum(wi * stats[d][0] for wi, d in zip(w, src))
        cov = sum(wi * (stats[d][1] + np.outer(stats[d][0], stats[d][0])) for wi, d in zip(w, src)) - np.outer(mu, mu)
        lodo[t] = round(frechet(mu, cov, *stats[t][:2]), 2)
    K.log(f"lodo FD: {lodo}")
    del model
    torch.cuda.empty_cache()
    o_pair = {a: {} for a in DOMAINS}
    for i, a in enumerate(DOMAINS):
        o_pair[a][a] = 0.0
        for b in DOMAINS[i + 1:]:
            o_pair[a][b] = o_pair[b][a] = round(otdd(*feats[a], *feats[b]), 2)
    o_lodo = {}
    for t in DOMAINS:
        src = [d for d in DOMAINS if d != t]
        o_lodo[t] = round(otdd(np.concatenate([feats[d][0] for d in src]), np.concatenate([feats[d][1] for d in src]),
                               *feats[t]), 2)
    K.log(f"lodo OTDD: {o_lodo}")
    K.save_json(K.RESULTS / "distance_dinov3.json", {
        "kind": "distance", "features": "DINOv3 ViT-S pretrained (384-d), 20 imgs/class of each train sample",
        "frechet": {"pairwise": pair, "lodo": lodo}, "otdd": {"pairwise": o_pair, "lodo": o_lodo, "n_points": N_OT},
        "per_class": PER_CLASS, "sec": round(time.time() - t0), "complete": True})


if __name__ == "__main__":
    main()
