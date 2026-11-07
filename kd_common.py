"""Shared pieces for the knowledge distillation (KD) and ERM experiments: data, model, training loop, evaluation.

Builds on common.py (DomainNet lists, backbone names, transforms). Images are read from DOMAINNET_ROOT,
as in the rest of the repo. Results go to results/*.json and checkpoints to models/*.pt.

Set KD_SMOKE=1 for a tiny end-to-end test (a few steps per job, outputs in logs/kd_smoke/).
"""
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Sampler  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from common import BACKBONES, BICUBIC, DOMAINS, NUM_CLASSES, ROOT, DomainNet  # noqa: E402

SMOKE = os.environ.get("KD_SMOKE") == "1"
SMOKE_STEPS = 6
BASE = REPO / "logs" / "kd_smoke" if SMOKE else REPO
RESULTS, MODELS = BASE / "results", BASE / "models"
for _p in (RESULTS, MODELS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- adaptation thresholds (defined in one place)
# Ceiling = in-domain accuracy of DINOv3 ViT-S fine-tuned on that domain with finetune.py
# (diagonal of the DINOv3 table in notes.md, full test set, %).
CEILING = {"clipart": 77.2, "infograph": 43.1, "painting": 71.7, "quickdraw": 63.3, "real": 80.3, "sketch": 67.5}
THRESHOLD_FRAC = 0.95                 # headline threshold
THRESHOLDS = (0.80, 0.90, 0.95)       # all reported


def threshold(target, frac=THRESHOLD_FRAC):
    return frac * CEILING[target]


# ---------------------------------------------------------------- training recipe (same as finetune.py)
STUDENT = "dinov3"
PER_CLASS, SAMPLE_SEED, VIEW_SEED, BIG = 50, 0, 0, 10_000_000
BATCH, EVAL_BATCH = 64, 256
EXPERT_EPOCHS = 10   # single-domain training, as in finetune.py
LODO_STEPS = 5000    # multi-source training length (DomainBed default, Gulrajani & Lopez-Paz 2021)
WORKERS = 2 if SMOKE else 6  # more workers ran out of memory on an 8 GB GPU machine
MEAN = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
STD = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
# same ops as common.TRAIN_TF / TEST_TF, but returning uint8 (normalised on the GPU)
TRAIN_U8 = T.Compose([T.RandomResizedCrop(224, (0.5, 1), interpolation=BICUBIC), T.RandomHorizontalFlip(), T.PILToTensor()])
TEST_U8 = T.Compose([T.Resize(256, BICUBIC), T.CenterCrop(224), T.PILToTensor()])


def norm(x_u8):
    return (x_u8.float() / 255.0 - MEAN.to(x_u8.device)) / STD.to(x_u8.device)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def load_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return default


# ---------------------------------------------------------------- data
def train_items():
    """All 6 domains' seeded 50/class train samples (same sample as finetune.py with seed 0), concatenated.
    Returns items [(path, label)] and the domain index of each item."""
    items, dom = [], []
    for di, d in enumerate(DOMAINS):
        it = DomainNet(d, "train", per_class=PER_CLASS, seed=SAMPLE_SEED).items
        if SMOKE:
            it = it[:: max(1, len(it) // 300)]
        items += it
        dom += [di] * len(it)
    return items, np.array(dom)


def test_items(d, limit=0, seed=0):
    if SMOKE:
        limit = 200
    return DomainNet(d, "test", limit=limit, seed=seed).items


class ViewDS(Dataset):
    """Item code = epoch * BIG + index. The augmented view is a fixed function of (view_seed, epoch, index),
    so every method sees exactly the same augmented images."""

    def __init__(self, items, view_seed=VIEW_SEED):
        self.items, self.view_seed = items, view_seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, code):
        e, i = divmod(int(code), BIG)
        path, y = self.items[i]
        torch.manual_seed(self.view_seed * 1_000_003 + e * BIG + i)
        return TRAIN_U8(Image.open(ROOT / path).convert("RGB")), y, i, e


class TestDS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        return TEST_U8(Image.open(ROOT / path).convert("RGB")), y


class EpochSampler(Sampler):
    def __init__(self, idx, epochs, seed=0, shuffle=True):
        self.idx, self.epochs, self.seed, self.shuffle = np.asarray(idx, np.int64), int(epochs), seed, shuffle

    def __len__(self):
        return self.epochs * len(self.idx)

    def __iter__(self):
        for e in range(self.epochs):
            order = np.random.default_rng([self.seed, e, 99]).permutation(len(self.idx)) if self.shuffle \
                else np.arange(len(self.idx))
            for j in order:
                yield int(e * BIG + self.idx[j])


def view_loader(items, idx, epochs, seed=0, shuffle=True, batch=BATCH, drop_last=True, view_seed=VIEW_SEED):
    return DataLoader(ViewDS(items, view_seed), batch_size=batch, sampler=EpochSampler(idx, epochs, seed, shuffle),
                      drop_last=drop_last, num_workers=WORKERS, pin_memory=True, prefetch_factor=4)


def test_loader(items):
    return DataLoader(TestDS(items), batch_size=EVAL_BATCH, num_workers=WORKERS, pin_memory=True, prefetch_factor=4)


# ---------------------------------------------------------------- model
def new_model(pretrained=True, name=None):
    return timm.create_model(name or BACKBONES[STUDENT], pretrained=pretrained, num_classes=NUM_CLASSES).cuda()


def load_init(model, path):
    """Loads any DINOv3 ViT-S state dict (plain, or nested under model/state_dict). Keys with a different shape
    (e.g. another head) are skipped and keep their fresh init. Returns a short report."""
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "model_state_dict"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    sd = {k.removeprefix("module.").removeprefix("_orig_mod."): v for k, v in sd.items()}
    own = model.state_dict()
    skipped = [k for k, v in sd.items() if k in own and own[k].shape != v.shape]
    sd = {k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}
    res = model.load_state_dict(sd, strict=False)
    rep = {"loaded": len(sd), "missing": list(res.missing_keys), "skipped_shape": skipped,
           "unexpected": list(res.unexpected_keys)}
    if len(rep["missing"]) > 2:
        raise RuntimeError(f"init {path} does not look like a DINOv3 ViT-S checkpoint: {rep['missing'][:5]}")
    return rep


def make_opt(model, total, warmup, lr=3e-5, head_lr=1e-3):
    head = list(model.get_classifier().parameters())
    hid = {id(p) for p in head}
    body = [p for p in model.parameters() if id(p) not in hid]
    opt = torch.optim.AdamW([{"params": body, "lr": lr}, {"params": head, "lr": head_lr}], weight_decay=0.05)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 0.5 * (1 + math.cos(math.pi * min(s, total) / total))))
    return opt, sched


def train(model, loader, total, warmup, loss_fn, teacher=None, tag="train", log_every=200, accum=1):
    """teacher(x_normalised, idx, epoch, y) -> teacher logits on the GPU (or None). accum: gradient accumulation
    (loader batch = BATCH / accum, one optimiser step per accum batches)."""
    if SMOKE:
        total, warmup = min(total, SMOKE_STEPS), 1
    opt, sched = make_opt(model, total, warmup)
    model.train()
    t0, step, k = time.time(), 0, 0
    opt.zero_grad(set_to_none=True)
    for x, y, i, e in loader:
        x, y = norm(x.cuda(non_blocking=True)), y.cuda(non_blocking=True)
        zt = teacher(x, i.numpy(), e.numpy(), y) if teacher else None
        with torch.autocast("cuda", torch.bfloat16):
            zs = model(x)
        loss, _ = loss_fn(zs.float(), zt, y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        (loss / accum).backward()
        k += 1
        if k % accum:
            continue
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % log_every == 0 or step == total or step == 20:
            r = step / (time.time() - t0)
            log(f"{tag} step {step}/{total} loss {loss.item():.3f} | {r:.2f} it/s | ~{(total - step) / r / 60:.0f} min left")
        if step >= total:
            break
    return {"steps": step, "train_sec": round(time.time() - t0, 1)}


def eval_report(model, heldout=None, own=None):
    """Full official test set for the held-out (or own) domain, 2,000-image subsets for the other domains."""
    items, dom = [], []
    for di, d in enumerate(DOMAINS):  # one loader for all domains (fewer worker processes)
        it = test_items(d, 0 if d in (heldout, own) else 2000, 0)
        items += it
        dom += [di] * len(it)
    dom = np.array(dom)
    hits = np.zeros(len(DOMAINS))
    model.eval()
    n0 = 0
    with torch.no_grad():
        for x, y in test_loader(items):
            with torch.autocast("cuda", torch.bfloat16):
                ok = (model(norm(x.cuda(non_blocking=True))).argmax(1).cpu() == y).numpy()
            np.add.at(hits, dom[n0:n0 + len(y)], ok)
            n0 += len(y)
    model.train()
    acc = {d: round(100.0 * hits[di] / max((dom == di).sum(), 1), 2) for di, d in enumerate(DOMAINS)}
    log(f"eval {acc}")
    return acc
