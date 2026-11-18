"""Training jobs for the KD and ERM experiments (DINOv3 ViT-S student). One job per process.

Train on one domain (the setting shared with the other methods):
  python kd_train.py expert      --domain sketch    # ERM: ViT-S fine-tuned on one domain (finetune.py recipe)
  python kd_train.py src-teacher --domain sketch    # ViT-B teacher fine-tuned on the same domain
  python kd_train.py src-student --domain sketch    # ViT-S student distilled from that teacher
  python kd_train.py src-student --domain sketch --T 4 --wkd 0.5   # KD hyperparameter sweep

Train on two domains (main setting):
  python kd_train.py pair-student --domains clipart sketch --T 4 --wkd 0.5
      # each image is distilled from the single-domain ViT-B teacher of its own domain

Quality reference:
  python kd_train.py all6-erm      # ERM on all six domains pooled (10 epochs)

Leave one domain out (train on the other five):
  python kd_train.py lodo-teacher --target sketch
  python kd_train.py lodo-student --method erm --teacher none    --target sketch
  python kd_train.py lodo-student --method kd  --teacher vitb    --target sketch
  python kd_train.py lodo-student --method kd  --teacher experts --target sketch

Single-domain jobs use exactly the images, epochs and schedule of `expert`, so KD and ERM differ only by the
teacher. Each writes results/<name>.json and models/<name>.pt.
"""
import argparse
import functools
import time

import numpy as np
import torch
import torch.nn.functional as F

import kd_common as K
from kd_common import DOMAINS
from kd_losses import LOSSES

VITB = "vit_base_patch16_dinov3.lvd1689m"


# ---------------------------------------------------------------- names (match the result files)
def expert_name(d, seed=0):
    return f"expert_dinov3_{d}_s{seed}"


def src_teacher_name(d, seed=0):
    return f"teacher-vitb_dinov3_src-{d}_s{seed}"


def src_student_name(d, seed=0):
    return f"kd-kd-vitb_dinov3_src-{d}_s{seed}"


def sweep_student_name(d, T, wkd, seed=0):
    return f"kdsweep-vitb_dinov3_src-{d}_T{T:g}_w{wkd:g}_s{seed}"


def pair_student_name(doms, T, wkd, seed=0):
    a, b = sorted(doms)
    return f"kd-vitb_dinov3_src-{a}+{b}_T{T:g}_w{wkd:g}_s{seed}"


def all6_name(seed=0):
    return f"erm_dinov3_all6_10ep_s{seed}"


def lodo_teacher_name(target, seed=0):
    return f"teacher-vitb_dinov3_lodo-{target}_5k_s{seed}"


def lodo_student_name(method, teacher, target, seed=0):
    pre = "erm" if method == "erm" else f"kd-{method}-{teacher}"
    return f"{pre}_dinov3_lodo-{target}_5k_s{seed}"


# ---------------------------------------------------------------- teachers
def frozen_vitb(path):
    tm = K.new_model(pretrained=False, name=VITB)
    K.load_init(tm, path)
    tm.eval().requires_grad_(False)

    def fn(x, i, e, y):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return tm(x).float()
    return fn


def source_experts_teacher(target, dom):
    """DAEL-style (Zhou et al. 2021): an image from source s is taught by the mean prediction of the experts of
    the other source domains, so no teacher has trained on that image."""
    srcs = [d for d in DOMAINS if d != target]
    ms = []
    for d in srcs:
        m = K.new_model(pretrained=False)
        K.load_init(m, K.MODELS / f"{expert_name(d)}.pt")
        ms.append(m.eval().requires_grad_(False))
    sid = torch.tensor([DOMAINS.index(d) for d in srcs], device="cuda")

    def fn(x, i, e, y):
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            P = torch.stack([F.softmax(m(x).float(), 1) for m in ms])
        msk = (sid.view(-1, 1) != torch.from_numpy(dom[i]).cuda().view(1, -1)).float().unsqueeze(-1)
        return ((P * msk).sum(0) / msk.sum(0)).clamp_min(1e-8).log()
    return fn


# ---------------------------------------------------------------- train on one domain
def single_source_run(a, model, name, kind, loss, teacher=None, accum=1, extra=None):
    t0 = time.time()
    torch.manual_seed(a.seed)
    items, dom = K.train_items()
    idx = np.nonzero(dom == DOMAINS.index(a.domain))[0]
    steps = K.EXPERT_EPOCHS * (len(idx) // K.BATCH)
    dl = K.view_loader(items, idx, K.EXPERT_EPOCHS, seed=a.seed, batch=K.BATCH // accum)
    info = K.train(model, dl, steps, len(idx) // K.BATCH, loss, teacher, tag=name, accum=accum)
    torch.save(model.state_dict(), K.MODELS / f"{name}.pt")
    acc = K.eval_report(model, own=a.domain)
    K.save_json(K.RESULTS / f"{name}.json", {
        "kind": kind, "domain": a.domain, "seed": a.seed, "in_domain": acc[a.domain], "acc": acc,
        "ceiling": K.CEILING[a.domain],
        "cross_mean": round(float(np.mean([acc[d] for d in DOMAINS if d != a.domain])), 2),
        "n_train": len(idx), **(extra or {}), **info, "sec": round(time.time() - t0), "complete": True})


def job_expert(a):
    torch.manual_seed(a.seed)  # seed before the head is initialised
    single_source_run(a, K.new_model(), expert_name(a.domain, a.seed), "expert", LOSSES["erm"])


def job_src_teacher(a):
    single_source_run(a, K.new_model(name=VITB), src_teacher_name(a.domain, a.seed), "src-teacher", LOSSES["erm"],
                      accum=a.accum, extra={"arch": VITB})


def job_src_student(a):
    """KD (Hinton) from the ViT-B teacher of the same domain. T = 1 / KD weight 0.5 is the main single-domain
    setting; other values are the hyperparameter sweep (CE weight = 1 - KD weight)."""
    teacher = frozen_vitb(K.MODELS / f"{src_teacher_name(a.domain)}.pt")
    if a.T == 1.0 and a.wkd == 0.5:
        single_source_run(a, K.new_model(), src_student_name(a.domain, a.seed), "src-student", LOSSES["kd"],
                          teacher, extra={"method": "kd", "teacher": "vitb"})
        return
    loss = functools.partial(LOSSES["kd"], T=a.T, w_ce=1.0 - a.wkd, w_kd=a.wkd)
    single_source_run(a, K.new_model(), sweep_student_name(a.domain, a.T, a.wkd, a.seed), "src-student", loss,
                      teacher, extra={"method": "kd", "teacher": "vitb", "T": a.T, "w_kd": a.wkd, "w_ce": 1.0 - a.wkd})


# ---------------------------------------------------------------- train on two domains
def job_pair_student(a):
    """ViT-S student on two pooled source domains (single-domain recipe: 10 epochs over both domains' images).
    Multi-teacher KD: each image is distilled from the single-domain ViT-B teacher of its own domain."""
    doms = sorted(a.domains)
    if len(doms) != 2 or doms[0] == doms[1]:
        raise SystemExit("--domains needs two different domains")
    t0 = time.time()
    torch.manual_seed(a.seed)
    teachers = {}
    for d in doms:
        tm = K.new_model(pretrained=False, name=VITB)
        K.load_init(tm, K.MODELS / f"{src_teacher_name(d)}.pt")
        teachers[DOMAINS.index(d)] = tm.eval().requires_grad_(False)
    items, dom = K.train_items()

    def teacher(x, i, e, y):
        di = torch.from_numpy(dom[i]).to(x.device)
        out = torch.empty(x.shape[0], K.NUM_CLASSES, device=x.device)
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for k, tm in teachers.items():
                m = di == k
                if m.any():
                    out[m] = tm(x[m]).float()
        return out

    loss = functools.partial(LOSSES["kd"], T=a.T, w_ce=1.0 - a.wkd, w_kd=a.wkd)
    model = K.new_model()
    name = pair_student_name(doms, a.T, a.wkd, a.seed)
    idx = np.nonzero(np.isin(dom, [DOMAINS.index(d) for d in doms]))[0]
    steps = K.EXPERT_EPOCHS * (len(idx) // K.BATCH)
    dl = K.view_loader(items, idx, K.EXPERT_EPOCHS, seed=a.seed)
    info = K.train(model, dl, steps, len(idx) // K.BATCH, loss, teacher, tag=name)
    torch.save(model.state_dict(), K.MODELS / f"{name}.pt")
    acc = K.eval_report(model, heldout=doms[1], own=doms[0])  # full test sets for both sources
    K.save_json(K.RESULTS / f"{name}.json", {
        "kind": "pair-student", "domains": doms, "seed": a.seed, "acc": acc,
        "source_mean": round(float(np.mean([acc[d] for d in doms])), 2),
        "cross_mean": round(float(np.mean([acc[d] for d in DOMAINS if d not in doms])), 2),
        "n_train": len(idx), "method": "kd", "teacher": "vitb-per-domain", "T": a.T, "w_kd": a.wkd,
        "w_ce": 1.0 - a.wkd, **info, "sec": round(time.time() - t0), "complete": True})


# ---------------------------------------------------------------- quality reference
@torch.no_grad()
def full_test_acc(model, d):
    model.eval()
    hits, n = 0, 0
    for x, y in K.test_loader(K.test_items(d)):
        with torch.autocast("cuda", torch.bfloat16):
            hits += (model(K.norm(x.cuda(non_blocking=True))).argmax(1).cpu() == y).sum().item()
        n += len(y)
    model.train()
    return round(100.0 * hits / max(n, 1), 2)


def job_all6_erm(a):
    """ERM on all six domains pooled (50 images per class and domain, 10 epochs). Accuracy on every domain's full
    test set, and on the 20-per-class test subset used by kd_adapt.py."""
    import kd_adapt as AD
    t0 = time.time()
    torch.manual_seed(a.seed)
    items, dom = K.train_items()
    idx = np.arange(len(items))
    steps = K.EXPERT_EPOCHS * (len(idx) // K.BATCH)
    name = all6_name(a.seed)
    model = K.new_model()
    dl = K.view_loader(items, idx, K.EXPERT_EPOCHS + 2, seed=a.seed)
    info = K.train(model, dl, steps, len(idx) // K.BATCH, LOSSES["erm"], tag=name)
    torch.save(model.state_dict(), K.MODELS / f"{name}.pt")
    full = {d: full_test_acc(model, d) for d in DOMAINS}
    sub = {d: AD.acc_on(model, *AD.eval_set(d, 20)) for d in DOMAINS}
    K.log(f"full test {full} | adaptation test subset {sub}")
    K.save_json(K.RESULTS / f"{name}.json", {
        "kind": "all6-erm", "seed": a.seed, "acc_full_test": full, "acc_adapt_subset_20pc": sub,
        "n_train": len(idx), **info, "sec": round(time.time() - t0), "complete": True})


# ---------------------------------------------------------------- leave one domain out
def lodo_run(a, model, name, kind, loss, teacher=None, accum=1, extra=None):
    t0 = time.time()
    torch.manual_seed(a.seed)
    items, dom = K.train_items()
    idx = np.nonzero(dom != DOMAINS.index(a.target))[0]
    steps = K.LODO_STEPS
    epochs = steps * K.BATCH // len(idx) + 2
    dl = K.view_loader(items, idx, epochs, seed=a.seed, batch=K.BATCH // accum)
    info = K.train(model, dl, steps, len(idx) // K.BATCH, loss, teacher, tag=name, accum=accum)
    torch.save(model.state_dict(), K.MODELS / f"{name}.pt")
    acc = K.eval_report(model, heldout=a.target)
    K.save_json(K.RESULTS / f"{name}.json", {
        "kind": kind, "target": a.target, "seed": a.seed, "heldout": acc[a.target], "acc": acc,
        "source_mean": round(float(np.mean([acc[d] for d in DOMAINS if d != a.target])), 2),
        "n_train": len(idx), **(extra or {}), **info, "sec": round(time.time() - t0), "complete": True})


def job_lodo_teacher(a):
    """DINOv3 ViT-B/16 fine-tuned on the 5 source domains with the student's recipe."""
    torch.manual_seed(a.seed)
    lodo_run(a, K.new_model(name=VITB), lodo_teacher_name(a.target, a.seed), "teacher", LOSSES["erm"],
             accum=a.accum, extra={"arch": VITB})


def job_lodo_student(a):
    torch.manual_seed(a.seed)
    _, dom = K.train_items()
    teacher = None
    if a.teacher == "vitb":
        teacher = frozen_vitb(K.MODELS / f"{lodo_teacher_name(a.target)}.pt")
    elif a.teacher == "experts":
        teacher = source_experts_teacher(a.target, dom)
    lodo_run(a, K.new_model(), lodo_student_name(a.method, a.teacher, a.target, a.seed), "student",
             LOSSES[a.method], teacher, extra={"method": a.method, "teacher": a.teacher, "steps_total": K.LODO_STEPS})


JOBS = {"expert": job_expert, "src-teacher": job_src_teacher, "src-student": job_src_student,
        "pair-student": job_pair_student, "all6-erm": job_all6_erm,
        "lodo-teacher": job_lodo_teacher, "lodo-student": job_lodo_student}

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("kind", choices=list(JOBS))
    p.add_argument("--domain", choices=DOMAINS, help="training domain (single-domain jobs)")
    p.add_argument("--domains", nargs=2, choices=DOMAINS, help="the two training domains (pair-student)")
    p.add_argument("--target", choices=DOMAINS, help="held-out domain (leave-one-out jobs)")
    p.add_argument("--method", default="erm", choices=list(LOSSES))
    p.add_argument("--teacher", default="none", choices=["none", "vitb", "experts"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--accum", type=int, default=1, help="gradient accumulation steps (ViT-B on 8 GB: 2)")
    p.add_argument("--T", type=float, default=1.0, help="KD temperature (src-student, pair-student)")
    p.add_argument("--wkd", type=float, default=0.5, help="KD loss weight; CE weight = 1 - wkd")
    a = p.parse_args()
    JOBS[a.kind](a)
