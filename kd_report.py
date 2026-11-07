"""Builds kd_results/RESULTS.md and its figures (PNG + SVG) from results/*.json.

  python kd_report.py [--results results] [--out kd_results] [--smoke]

Safe to run at any time: experiments that have not finished yet are shown as pending.
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

REPO = Path(__file__).resolve().parent
D = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
CEILING = {"clipart": 77.2, "infograph": 43.1, "painting": 71.7, "quickdraw": 63.3, "real": 80.3, "sketch": 67.5}
FR = ("0.8", "0.9", "0.95")
PENDING = "pending"

# Colour follows the method, never its rank. Categorical slots 1-3 pass the CVD / contrast checks;
# grey marks the reference baseline (no DomainNet training).
C_ERM, C_KD, C_KDX, C_BASE = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a85"
INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
LODO_STUD = [("erm", "ERM", C_ERM), ("kd-kd-experts", "KD, source-expert teachers", C_KDX),
             ("kd-kd-vitb", "KD, ViT-B teacher", C_KD)]
LODO_INIT = [("dinov3-lpft", "DINOv3, no DomainNet training (LP-FT)", C_BASE), ("erm-5k", "ERM", C_ERM),
             ("kd-experts-5k", "KD, source-expert teachers", C_KDX), ("kd-vitb-5k", "KD, ViT-B teacher", C_KD)]
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ef9a99", "#c62f2f"])

plt.rcParams.update({
    "font.size": 12, "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": AXIS, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "axes.axisbelow": True, "axes.titleweight": "bold", "axes.titlesize": 12, "lines.linewidth": 2,
    "figure.facecolor": "white", "axes.facecolor": "white", "legend.frameon": False, "savefig.facecolor": "white"})


# ---------------------------------------------------------------- loading
def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


class Results:
    def __init__(self, res_dir):
        self.res = res_dir

    def get(self, name):
        r = load(self.res / f"{name}.json")
        return r if r.get("complete") else {}

    def lodo(self, key, t):
        return self.get(f"{key}_dinov3_lodo-{t}_5k_s0").get("heldout")

    def adapt(self, short, t):
        """Seed 0 plus seeds 1-2 when present; statistics recomputed over all seeds.
        Seeds re-draw the 10 fine-tuning images per class; the starting model is the same."""
        r = self.get(f"adapt_{short}_to-{t}_k10_v2")
        if not r:
            return {}
        r12 = self.get(f"adapt_{short}_to-{t}_k10_v2_s12")
        R = sorted(r["runs"] + (r12["runs"] if r12 else []), key=lambda x: x["seed"])
        cs = np.array([[a for _, a in x["curve"]] for x in R])
        aucs = [x["auc"] for x in R]

        def med(f):  # median over seeds; a seed that never reaches the threshold counts as infinite
            v = float(np.median([x["steps_to"][f] if x["steps_to"][f] is not None else np.inf for x in R]))
            return None if np.isinf(v) else (int(v) if v == int(v) else v)
        return {**r, "runs": R, "n_seeds": len(R),
                "mean_curve": [[s, float(v)] for (s, _), v in zip(R[0]["curve"], cs.mean(0))],
                "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs, ddof=1)) if len(R) > 1 else 0.0,
                "start_mean": float(np.mean([x["start"] for x in R])), "steps_med": {f: med(f) for f in FR}}


# ---------------------------------------------------------------- statistics
def spearman_perm(x, y, n=20000):
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(rx, rng.permutation(ry))[0, 1] for _ in range(n)])
    return rho, float((np.abs(perm) >= abs(rho) - 1e-12).mean())


def mantel(pairs, dist, ys):
    """Spearman rho over (source, target) pairs. Exact p-value from all 720 relabelings of the 6 domains
    (Mantel test), because pairs that share a domain are not independent."""
    ry = rankdata(ys)
    rho = float(np.corrcoef(rankdata([dist[s][t] for s, t in pairs]), ry)[0, 1])
    null = []
    for p in itertools.permutations(range(6)):
        m = {D[i]: D[p[i]] for i in range(6)}
        null.append(np.corrcoef(rankdata([dist[m[s]][m[t]] for s, t in pairs]), ry)[0, 1])
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean())


def paired(a, b):
    """Per-seed AUC differences a - b for two adaptation results on the same target (same k-shot samples)."""
    if not (a and b):
        return []
    bs = {x["seed"]: x["auc"] for x in b["runs"]}
    return [x["auc"] - bs[x["seed"]] for x in a["runs"] if x["seed"] in bs]


def diff_summary(d):
    if not d:
        return PENDING
    return f"{np.mean(d):+.2f} (range {min(d):+.2f} to {max(d):+.2f}; higher in {sum(x > 0 for x in d)} of {len(d)})"


def f1(x, fmt="{:.1f}"):
    return PENDING if x is None else fmt.format(x)


def pval(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def table(header, rows, numeric=True):
    align = ["---"] + (["--:"] if numeric else ["---"]) * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    return out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def save(fig, out, name):
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- sections
def setup_section(R):
    ex = R.get("expert_dinov3_sketch_s0")
    check = (f" As a pipeline check, the single-domain ERM model for sketch reaches {ex['in_domain']:.1f}% "
             f"on the sketch test set (finetune.py: 67.5%).") if ex else ""
    rows = [
        ("Student", "DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`, 21.6M parameters)"),
        ("Teacher", "DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`), fine-tuned on the same source data "
                    "with the same recipe, then frozen"),
        ("Training data", "50 images per class and domain (seeded sample, identical to finetune.py), 345 classes"),
        ("Training recipe", "finetune.py: AdamW (backbone lr 3e-5, head lr 1e-3, weight decay 0.05), batch 64, "
                            "1-epoch warm-up then cosine decay, bf16"),
        ("Training length", "one source domain: 10 epochs (finetune.py); five source domains: 5,000 steps "
                            "(DomainBed default)"),
        ("ERM", "cross-entropy on the source data"),
        ("KD", "Hinton et al. (2015), T = 1, 0.5 cross-entropy + 0.5 KL divergence to the teacher"),
        ("Baseline", "pretrained DINOv3 without DomainNet training. Its classifier head is first fitted on the "
                     "10-shot data with the backbone frozen, then the whole model is fine-tuned (LP-FT, Kumar et al. "
                     "2022); a randomly initialised head would put it at an unfair disadvantage. ERM and KD models "
                     "keep their trained head."),
        ("Adaptation test", "fine-tune on 10 labelled images per class of the target domain for 500 steps; "
                            "accuracy on a fixed class-balanced subset of the target test set (20 images per class)"),
        ("Adaptation metrics", "steps to reach 80%, 90% and 95% of the target's in-domain ceiling; "
                               "AUC = mean accuracy over the 500 steps"),
        ("Ceiling", "in-domain accuracy of DINOv3 ViT-S fine-tuned on the target domain (notes.md)"),
        ("Repetitions", "each model is trained once; each adaptation test is repeated with 3 different "
                        "10-shot samples"),
    ]
    return ["## 1. Experimental setup", ""] + table(["Item", "Setting"], rows, numeric=False) + ["", check.strip(), ""]


def single_source_section(R, out, figs):
    md = ["## 2. Single-source transfer (train on one domain, adapt to the other five)", "",
          "Each model is trained on one source domain and then adapted to each of the five other domains. "
          "The ERM model is the single-domain model trained with finetune.py's recipe; the KD model is trained on "
          "the same images for the same number of steps, with a ViT-B teacher trained on that domain.", "",
          "### 2.1 Accuracy before adaptation", "",
          "Top-1 accuracy (%) on the source domain's full test set, and the mean over the other five domains "
          "(2,000 test images each), before any target data is used.", ""]
    rows = []
    for s in D:
        e, k, t = (R.get(f"expert_dinov3_{s}_s0"), R.get(f"kd-kd-vitb_dinov3_src-{s}_s0"),
                   R.get(f"teacher-vitb_dinov3_src-{s}_s0"))
        cross = lambda r: None if not r else float(np.mean([r["acc"][d] for d in D if d != s]))
        rows.append([s, f1(e.get("in_domain")), f1(cross(e)), f1(k.get("in_domain")), f1(cross(k)),
                     f1(t.get("in_domain")), f1(cross(t))])
    md += table(["Source", "ERM, source", "ERM, other five", "KD, source", "KD, other five",
                 "ViT-B teacher, source", "ViT-B teacher, other five"], rows) + [""]

    md += ["### 2.2 Adaptation to each target", "",
           "Mean AUC over the five targets and three 10-shot samples, and the paired difference KD − ERM "
           "(same 10-shot samples for both models).", ""]
    rows, all_d, grid = [], [], np.full((6, 6), np.nan)
    for i, s in enumerate(D):
        a_e, a_k, d_s = [], [], []
        for j, t in enumerate(D):
            if t == s:
                continue
            e, k = R.adapt(f"expert-{s}", t), R.adapt(f"kd-vitb-src-{s}", t)
            if e:
                a_e.append(e["auc_mean"])
            if k:
                a_k.append(k["auc_mean"])
            d = paired(k, e)
            if d:
                d_s += d
                grid[i, j] = np.mean(d)
        all_d += d_s
        rows.append([s, f1(np.mean(a_e) if len(a_e) == 5 else None), f1(np.mean(a_k) if len(a_k) == 5 else None),
                     diff_summary(d_s) if len(d_s) == 15 else (PENDING if not d_s else f"{len(d_s)} of 15 complete")])
    md += table(["Source", "ERM, mean AUC", "KD, mean AUC", "KD − ERM, AUC points"], rows)
    md += ["", f"**All pairs:** KD − ERM = {diff_summary(all_d)} over {len(all_d)} of 90 "
               "(source, target, sample) comparisons. Comparisons that share a model are not independent.", ""]
    if np.isfinite(grid).any():
        lim = max(0.5, float(np.nanmax(np.abs(grid))))
        fig, ax = plt.subplots(figsize=(7.6, 6.2))
        ax.grid(False)
        im = ax.imshow(grid, cmap=DIVERGING, vmin=-lim, vmax=lim)
        for i, j in itertools.product(range(6), range(6)):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", color=INK2)
            elif np.isfinite(grid[i, j]):
                v = grid[i, j]
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=11,
                        color="white" if abs(v) > 0.6 * lim else INK)
        ax.set_xticks(range(6), D, rotation=30, ha="right")
        ax.set_yticks(range(6), D)
        ax.set_xlabel("Target domain")
        ax.set_ylabel("Source domain (training)")
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("KD − ERM, adaptation AUC (points)")
        cb.outline.set_visible(False)
        ax.set_title("KD − ERM adaptation AUC, single-source transfer\n"
                     "Red: KD adapts better; blue: ERM adapts better. Mean of 3 samples.", loc="left")
        save(fig, out, "fig1_single_source_kd_vs_erm")
        figs.append(("fig1_single_source_kd_vs_erm", "Paired difference in adaptation AUC between the KD and ERM "
                     "models for every (source, target) pair, averaged over three 10-shot samples."))
    return md, all_d


def lodo_section(R, out, figs):
    md = ["## 3. Leave-one-domain-out (train on five domains, adapt to the sixth)", "",
          "Each model is trained on five domains; the sixth is held out. Two KD teachers are compared: "
          "the single-domain models of the five sources (each image taught by the experts of the other "
          "sources, as in DAEL, Zhou et al. 2021) and a DINOv3 ViT-B trained on the same five domains.", "",
          "### 3.1 Held-out accuracy", "",
          "Top-1 accuracy (%) on the held-out domain's full test set, before adaptation.", ""]
    rows = []
    for t in D:
        tv = R.get(f"teacher-vitb_dinov3_lodo-{t}_5k_s0").get("heldout")
        rows.append([t] + [f1(R.lodo(k, t)) for k, _, _ in LODO_STUD] + [f1(tv), f"{CEILING[t]:.1f}"])
    md += table(["Held-out domain"] + [n for _, n, _ in LODO_STUD] + ["ViT-B teacher", "Ceiling"], rows) + [""]
    if any(R.lodo(k, t) is not None for k, _, _ in LODO_STUD for t in D):
        fig, ax = plt.subplots(figsize=(11, 4.8))
        w = 0.8 / len(LODO_STUD)
        for k, (key, name, col) in enumerate(LODO_STUD):
            ax.bar([i + (k - (len(LODO_STUD) - 1) / 2) * w for i in range(6)], [R.lodo(key, t) or 0 for t in D],
                   w - 0.02, color=col, label=name)
        for i, t in enumerate(D):
            ax.hlines(CEILING[t], i - .42, i + .42, colors=INK, linestyles="--", lw=1.2,
                      label="In-domain ceiling" if i == 0 else None)
        ax.set_xticks(range(6), D)
        ax.set_ylabel("Top-1 accuracy on held-out domain (%)")
        ax.set_xlabel("Held-out domain (trained on the other five)")
        ax.set_title("Held-out accuracy of ERM and KD students (DINOv3 ViT-S)", loc="left")
        ax.legend(ncol=4, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.16))
        save(fig, out, "fig2_lodo_heldout_accuracy")
        figs.append(("fig2_lodo_heldout_accuracy", "Accuracy on the held-out domain before adaptation. Dashed lines "
                     "mark the in-domain ceiling. Each model was trained once."))

    md += ["### 3.2 Adaptation to the held-out domain", "",
           "Each cell: accuracy at step 0 / median steps to reach 80, 90 and 95% of the ceiling (– = not reached "
           "within 500 steps) / AUC ± standard deviation over three 10-shot samples.", ""]
    rows = []
    for t in D:
        cells = []
        for short, _, _ in LODO_INIT:
            r = R.adapt(short, t)
            if not r:
                cells.append(PENDING)
                continue
            st = r["steps_med"]
            cells.append(f"{r['start_mean']:.1f} / " + ", ".join("–" if st.get(f) is None else str(st[f]) for f in FR)
                         + f" / {r['auc_mean']:.1f} ± {r['auc_sd']:.1f}")
        rows.append([t] + cells)
    md += table(["Held-out domain"] + [n for _, n, _ in LODO_INIT], rows)
    md += ["", "Paired AUC difference against ERM (same 10-shot samples), over 6 held-out domains × 3 samples. "
               "The 18 comparisons come from 6 trained models per method, so they are not independent.", ""]
    diffs = {}
    for s, name, _ in LODO_INIT:
        if s == "erm-5k":
            continue
        d = sum((paired(R.adapt(s, t), R.adapt("erm-5k", t)) for t in D), [])
        diffs[s] = d
        md.append(f"- {name} − ERM: {diff_summary(d)}")
    md.append("")
    curves = {(s, t): R.adapt(s, t).get("mean_curve") for s, _, _ in LODO_INIT for t in D}
    if any(curves.values()):
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        for ax, t in zip(axs.flat, D):
            for s, name, col in LODO_INIT:
                c = curves[(s, t)]
                if c:
                    ax.plot([a for a, _ in c], [b for _, b in c], color=col, label=name)
            ax.axhline(CEILING[t] * .9, color=INK, ls="--", lw=1, label="90% of in-domain ceiling")
            ax.set_title(t, loc="left")
        for ax in axs[1]:
            ax.set_xlabel("Fine-tuning steps on the target (batch 64)")
        for ax in axs[:, 0]:
            ax.set_ylabel("Target accuracy (%)")
        h, lab = max((ax.get_legend_handles_labels() for ax in axs.flat), key=lambda x: len(x[1]))
        fig.legend(h, lab, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.05), fontsize=10)
        fig.suptitle("Adaptation curves with 10 labelled images per class (mean of 3 samples)", x=0.01, ha="left",
                     fontweight="bold")
        fig.tight_layout(rect=(0, 0.03, 1, 0.97))
        save(fig, out, "fig3_lodo_adaptation_curves")
        figs.append(("fig3_lodo_adaptation_curves", "Accuracy on the held-out domain during fine-tuning, one panel "
                     "per held-out domain. The dashed line marks 90% of the in-domain ceiling."))
    return md, diffs


def distance_section(R, out, figs):
    dist = R.get("distance_dinov3")
    if not dist.get("otdd"):
        return ["## 4. Domain distance and adaptation", "", "Pending: distances not computed yet.", ""]
    O, Fd = dist["otdd"], dist["frechet"]
    md = ["## 4. Domain distance and adaptation", "",
          "Distances are computed on frozen, pretrained DINOv3 ViT-S features of 20 training images per class: "
          "the Fréchet distance between Gaussian fits of the two feature distributions, and OTDD "
          "(Alvarez-Melis & Fusi 2020) with a Gaussian approximation of the label-to-label cost.", "",
          "Pooled five source domains → held-out domain:", ""]
    md += table(["Held-out domain", "OTDD", "Fréchet"], [[t, f"{O['lodo'][t]:.1f}", f"{Fd['lodo'][t]:.0f}"] for t in D])
    md += ["", "Spearman correlation between distance and adaptation AUC (a negative ρ means that more distant "
               "targets adapt worse). For single-source pairs, p-values come from an exact Mantel test over all 720 "
               "relabelings of the six domains; for the six held-out domains, from a permutation test.", ""]
    rows, res = [], {}
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.4))
    ax = axs[0]
    for short, name, col in (("expert-{}", "ERM", C_ERM), ("kd-vitb-src-{}", "KD, ViT-B teacher", C_KD)):
        pairs = [(s, t) for s in D for t in D if s != t and R.adapt(short.format(s), t)]
        if len(pairs) < 6:
            continue
        auc = [R.adapt(short.format(s), t)["auc_mean"] for s, t in pairs]
        for metric, M in (("OTDD", O["pairwise"]), ("Fréchet", Fd["pairwise"])):
            r, p = mantel(pairs, M, auc)
            res[(name, metric)] = (r, p)
            rows.append([f"Single source, {name}", metric, len(pairs), f"{r:.2f}", pval(p)])
        r, p = res[(name, "OTDD")]
        ax.scatter([O["pairwise"][s][t] for s, t in pairs], auc, s=60, color=col, edgecolor="white", lw=1.5,
                   zorder=3, label=f"{name} (ρ = {r:.2f}, p = {pval(p)}, n = {len(pairs)})")
    ax.set_xlabel("OTDD, source domain → target domain")
    ax.set_ylabel("Adaptation AUC (mean accuracy, %)")
    ax.set_title("Single-source pairs", loc="left")
    if ax.collections:
        ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    ax = axs[1]
    for k, ((s, name, col), mk) in enumerate(zip(LODO_INIT, "osD^")):
        pts = [(O["lodo"][t], R.adapt(s, t)["auc_mean"]) for t in D if R.adapt(s, t)]
        if len(pts) >= 4:
            r, p = spearman_perm(*zip(*pts))
            rows.append([f"Leave-one-out, {name}", "OTDD", len(pts), f"{r:.2f}", pval(p)])
            dx = (k - 1.5) * 0.035  # dodge: methods share x, so offset them slightly for visibility
            ax.scatter([x + dx for x, _ in pts], [y for _, y in pts], s=64, marker=mk, color=col,
                       edgecolor="white", lw=1.2, zorder=3, label=f"{name} (ρ = {r:.2f}, p = {pval(p)})")
    ax.set_xlabel("OTDD, pooled five sources → held-out domain")
    ax.set_ylabel("Adaptation AUC (mean accuracy, %)")
    ax.set_title("Leave-one-domain-out, six held-out domains", loc="left")
    if ax.collections:
        ax.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.suptitle("Domain distance and adaptation AUC", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, out, "fig4_distance_vs_adaptation")
    figs.append(("fig4_distance_vs_adaptation", "Adaptation AUC against OTDD. Left: one point per (source, target) "
                 "pair. Right: one point per held-out domain and starting model; points are offset horizontally by up to "
                 "0.05 so that overlapping methods stay visible."))
    gains = [(O["lodo"][t], R.lodo("kd-kd-vitb", t) - R.lodo("erm", t)) for t in D
             if R.lodo("kd-kd-vitb", t) is not None and R.lodo("erm", t) is not None]
    if len(gains) >= 4:
        r, p = spearman_perm(*zip(*gains))
        rows.append(["Leave-one-out, KD (ViT-B) − ERM held-out accuracy", "OTDD", len(gains), f"{r:.2f}", pval(p)])
    md += table(["Comparison", "Distance", "n", "Spearman ρ", "p"], rows, numeric=False) + [""]
    return md, res


def summary_section(R, ss_d, lodo_d, dres):
    lines = ["## Summary", ""]
    if ss_d:
        lines.append(f"- **Single-source transfer:** KD with a ViT-B teacher differs from ERM in adaptation AUC by "
                     f"{np.mean(ss_d):+.2f} points on average (higher in {sum(x > 0 for x in ss_d)} "
                     f"of {len(ss_d)} comparisons{'' if len(ss_d) == 90 else '; experiment still running'}).")
    else:
        lines.append("- **Single-source transfer:** KD results pending.")
    d = lodo_d.get("kd-vitb-5k", [])
    if d:
        lines.append(f"- **Leave-one-domain-out:** KD with a ViT-B teacher differs from ERM in adaptation AUC by {np.mean(d):+.2f} "
                     f"points (higher in {sum(x > 0 for x in d)} of {len(d)} comparisons, from 6 models × 3 samples); KD with "
                     f"source-expert teachers by {np.mean(lodo_d.get('kd-experts-5k') or [np.nan]):+.2f}.")
    d = lodo_d.get("dinov3-lpft", [])
    if d:
        better = [t for t in D if np.mean(paired(R.adapt("dinov3-lpft", t), R.adapt("erm-5k", t)) or [0]) > 0]
        exc = (f", except on {', '.join(better)}, where the pretrained model adapts better" if better else "")
        lines.append(f"- **Training on DomainNet sources:** ERM adapts {-np.mean(d):.1f} AUC points better on average "
                     f"than pretrained DINOv3 without DomainNet training{exc}.")
    if ("ERM", "Fréchet") in dres:
        (r1, p1), (r2, p2) = dres[("ERM", "Fréchet")], dres[("ERM", "OTDD")]
        sig = [m for m, p in (("Fréchet", p1), ("OTDD", p2)) if p < 0.05]
        verdict = (f"significant at p < 0.05 for {' and '.join(sig)} only" if 0 < len(sig) < 2 else
                   "significant at p < 0.05 for both distances" if sig else "not significant at p < 0.05")
        lines.append(f"- **Distance:** for single-source ERM pairs, adaptation AUC decreases with distance "
                     f"(Fréchet ρ = {r1:.2f}, p = {pval(p1)}; OTDD ρ = {r2:.2f}, p = {pval(p2)}; n = 30); "
                     f"{verdict}. Leave-one-domain-out correlations use n = 6 and are reported in Section 4.")
    return lines + [""]


LIMITATIONS = [
    "## 5. Limitations", "",
    "- Each model was trained once. The three repetitions re-sample the 10-shot fine-tuning data only, so "
    "differences in held-out accuracy below about one point have no error estimate.",
    "- Infograph and quickdraw have the lowest ceilings and are also the most distant domains, so the distance "
    "correlation partly reflects target difficulty.",
    "- With six domains, the leave-one-domain-out analysis has n = 6 per method. Paired comparisons that reuse "
    "the same trained model (several targets or 10-shot samples) are not independent, so counts such as "
    "\"higher in 18 of 18\" overstate the evidence if read as independent trials.",
    "- KD and ERM students train for the same number of steps on the same images, but KD additionally requires "
    "training the ViT-B teacher, so its total compute is higher.", ""]

REPRODUCE = [
    "## Reproducing these results", "",
    "Requirements beyond requirements.txt: `scipy`, `matplotlib`, `POT` (optimal transport, for OTDD) and "
    "`numpy>=2`. DomainNet must be unpacked at `DOMAINNET_ROOT`, as for finetune.py.", "",
    "```bash",
    "python kd_run_all.py --dry-run   # list the jobs that still need to run",
    "python kd_run_all.py             # run everything (resumable); logs in logs/kd/",
    "python kd_report.py              # rebuild kd_results/RESULTS.md and the figures",
    "```", "",
    "Single jobs can be run directly, e.g. `python kd_train.py src-student --domain sketch` or "
    "`python kd_adapt.py --init models/expert_dinov3_real_s0.pt --target sketch`. "
    "Approximate cost on one RTX 3080 Laptop GPU (8 GB): about 25 GPU-hours for the single-source experiments "
    "(Section 2) and about 28 GPU-hours for leave-one-domain-out (Section 3).", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=REPO / "results")
    ap.add_argument("--out", type=Path, default=REPO / "kd_results")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.results, a.out = REPO / "logs" / "kd_smoke" / "results", REPO / "logs" / "kd_smoke" / "kd_results"
    a.out.mkdir(parents=True, exist_ok=True)
    R, figs = Results(a.results), []
    ss_md, ss_d = single_source_section(R, a.out, figs)
    lodo_md, lodo_d = lodo_section(R, a.out, figs)
    dist_md, dres = distance_section(R, a.out, figs)
    md = ["# Knowledge distillation and ERM on DomainNet", "",
          "Adaptation speed of DINOv3 ViT-S models trained with ERM or knowledge distillation (KD), measured as "
          "accuracy on a new domain during few-shot fine-tuning.", ""]
    md += summary_section(R, ss_d, lodo_d, dres) + setup_section(R) + ss_md + lodo_md + dist_md + LIMITATIONS + REPRODUCE
    if figs:
        md += ["## Figures", ""] + [f"- **{n}** (`{n}.png`, `{n}.svg`): {c}" for n, c in figs] + [""]
    md += ["_Generated by kd_report.py from results/*.json._"]
    (a.out / "RESULTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
