import argparse
import json
from pathlib import Path

import ot
import torch
from torch.utils.data import DataLoader

from common import DOMAINS, NUM_CLASSES, DomainNet, backbone

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="resnet")
p.add_argument("--dim", type=int, default=64)
p.add_argument("--samples", type=int, default=4000)
p.add_argument("--seeds", type=int, default=3)
args = p.parse_args()

model = backbone(args.backbone).eval()


@torch.no_grad()
def features(domain):
    X, y = [], []
    for x, t in DataLoader(DomainNet(domain, "train", per_class=100), batch_size=256, num_workers=10):
        with torch.autocast("cuda", torch.bfloat16):
            X.append(model(x.cuda()).float().cpu())
        y.append(t)
    return torch.cat(X).double(), torch.cat(y)


def sqrtm(S):
    w, Q = torch.linalg.eigh(S)
    return Q @ torch.diag_embed(w.clamp(min=0).sqrt()) @ Q.transpose(-1, -2)


def label_distances(a, b):
    (mu1, cov1), (mu2, cov2) = a, b
    root1 = sqrtm(cov1)
    trace2 = cov2.diagonal(dim1=-2, dim2=-1).sum(-1)
    W = torch.stack([cov1[i].trace() + trace2 - 2 * sqrtm(root1[i] @ cov2 @ root1[i]).diagonal(dim1=-2, dim2=-1).sum(-1) for i in range(NUM_CLASSES)])
    return torch.cdist(mu1, mu2) ** 2 + W.clamp(min=0)


data = {d: features(d) for d in DOMAINS}
X = torch.cat([X for X, _ in data.values()])
mean = X.mean(0)
V = torch.pca_lowrank(X - mean, q=args.dim, center=False)[2]
data = {d: ((X - mean) @ V, y) for d, (X, y) in data.items()}


def gaussians(X, y):
    return torch.stack([X[y == c].mean(0) for c in range(NUM_CLASSES)]), torch.stack([torch.cov(X[y == c].T, correction=0) for c in range(NUM_CLASSES)])


dist = {f"{a}->{b}": [] for a in DOMAINS for b in DOMAINS}
for seed in range(args.seeds):
    g = torch.Generator().manual_seed(seed)
    halves = {}
    for d, (X, y) in data.items():
        perm = torch.randperm(len(y), generator=g)
        halves[d] = [(X[h], y[h], gaussians(X[h], y[h])) for h in (perm[::2], perm[1::2])]
    for a in DOMAINS:
        for b in DOMAINS:
            (Xa, ya, sa), (Xb, yb, sb) = halves[a][0], halves[b][1]
            ia, ib = torch.randperm(len(ya), generator=g)[:args.samples], torch.randperm(len(yb), generator=g)[:args.samples]
            cost = torch.cdist(Xa[ia], Xb[ib]) ** 2 + label_distances(sa, sb)[ya[ia]][:, yb[ib]]
            w = torch.full((len(ia),), 1 / len(ia), dtype=torch.float64)
            dist[f"{a}->{b}"].append(ot.emd2(w, torch.full((len(ib),), 1 / len(ib), dtype=torch.float64), cost, numItermax=10**7).sqrt().item())
            print(f"seed {seed} {a} -> {b}: {dist[f'{a}->{b}'][-1]:.2f}", flush=True)

Path("results").mkdir(exist_ok=True)
Path(f"results/otdd_{args.backbone}.json").write_text(json.dumps(dist, indent=2))
