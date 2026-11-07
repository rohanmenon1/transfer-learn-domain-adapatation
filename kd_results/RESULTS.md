# Knowledge distillation and ERM on DomainNet

Adaptation speed of DINOv3 ViT-S models trained with ERM or knowledge distillation (KD), measured as accuracy on a new domain during few-shot fine-tuning.

## Summary

- **Single-source transfer:** KD results pending.
- **Leave-one-domain-out:** KD with a ViT-B teacher differs from ERM in adaptation AUC by +0.32 points (higher in 18 of 18 comparisons, from 6 models × 3 samples); KD with source-expert teachers by +0.04.
- **Training on DomainNet sources:** ERM adapts 8.4 AUC points better on average than pretrained DINOv3 without DomainNet training, except on quickdraw, where the pretrained model adapts better.
- **Distance:** for single-source ERM pairs, adaptation AUC decreases with distance (Fréchet ρ = -0.70, p = 0.022; OTDD ρ = -0.58, p = 0.076; n = 30); significant at p < 0.05 for Fréchet only. Leave-one-domain-out correlations use n = 6 and are reported in Section 4.

## 1. Experimental setup

| Item | Setting |
|---|---|
| Student | DINOv3 ViT-S/16 (`vit_small_patch16_dinov3.lvd1689m`, 21.6M parameters) |
| Teacher | DINOv3 ViT-B/16 (`vit_base_patch16_dinov3.lvd1689m`), fine-tuned on the same source data with the same recipe, then frozen |
| Training data | 50 images per class and domain (seeded sample, identical to finetune.py), 345 classes |
| Training recipe | finetune.py: AdamW (backbone lr 3e-5, head lr 1e-3, weight decay 0.05), batch 64, 1-epoch warm-up then cosine decay, bf16 |
| Training length | one source domain: 10 epochs (finetune.py); five source domains: 5,000 steps (DomainBed default) |
| ERM | cross-entropy on the source data |
| KD | Hinton et al. (2015), T = 1, 0.5 cross-entropy + 0.5 KL divergence to the teacher |
| Baseline | pretrained DINOv3 without DomainNet training. Its classifier head is first fitted on the 10-shot data with the backbone frozen, then the whole model is fine-tuned (LP-FT, Kumar et al. 2022); a randomly initialised head would put it at an unfair disadvantage. ERM and KD models keep their trained head. |
| Adaptation test | fine-tune on 10 labelled images per class of the target domain for 500 steps; accuracy on a fixed class-balanced subset of the target test set (20 images per class) |
| Adaptation metrics | steps to reach 80%, 90% and 95% of the target's in-domain ceiling; AUC = mean accuracy over the 500 steps |
| Ceiling | in-domain accuracy of DINOv3 ViT-S fine-tuned on the target domain (notes.md) |
| Repetitions | each model is trained once; each adaptation test is repeated with 3 different 10-shot samples |

As a pipeline check, the single-domain ERM model for sketch reaches 67.6% on the sketch test set (finetune.py: 67.5%).

## 2. Single-source transfer (train on one domain, adapt to the other five)

Each model is trained on one source domain and then adapted to each of the five other domains. The ERM model is the single-domain model trained with finetune.py's recipe; the KD model is trained on the same images for the same number of steps, with a ViT-B teacher trained on that domain.

### 2.1 Accuracy before adaptation

Top-1 accuracy (%) on the source domain's full test set, and the mean over the other five domains (2,000 test images each), before any target data is used.

| Source | ERM, source | ERM, other five | KD, source | KD, other five | ViT-B teacher, source | ViT-B teacher, other five |
|---|--:|--:|--:|--:|--:|--:|
| clipart | 77.4 | 41.8 | 77.8 | 41.6 | 83.4 | 50.0 |
| infograph | 43.5 | 42.3 | pending | pending | pending | pending |
| painting | 71.7 | 38.8 | pending | pending | pending | pending |
| quickdraw | 63.4 | 22.9 | pending | pending | pending | pending |
| real | 80.5 | 37.7 | pending | pending | pending | pending |
| sketch | 67.6 | 45.0 | pending | pending | pending | pending |

### 2.2 Adaptation to each target

Mean AUC over the five targets and three 10-shot samples, and the paired difference KD − ERM (same 10-shot samples for both models).

| Source | ERM, mean AUC | KD, mean AUC | KD − ERM, AUC points |
|---|--:|--:|--:|
| clipart | 56.3 | pending | pending |
| infograph | 59.0 | pending | pending |
| painting | 55.8 | pending | pending |
| quickdraw | 49.8 | pending | pending |
| real | 53.7 | pending | pending |
| sketch | 57.8 | pending | pending |

**All pairs:** KD − ERM = pending over 0 of 90 (source, target, sample) comparisons. Comparisons that share a model are not independent.

## 3. Leave-one-domain-out (train on five domains, adapt to the sixth)

Each model is trained on five domains; the sixth is held out. Two KD teachers are compared: the single-domain models of the five sources (each image taught by the experts of the other sources, as in DAEL, Zhou et al. 2021) and a DINOv3 ViT-B trained on the same five domains.

### 3.1 Held-out accuracy

Top-1 accuracy (%) on the held-out domain's full test set, before adaptation.

| Held-out domain | ERM | KD, source-expert teachers | KD, ViT-B teacher | ViT-B teacher | Ceiling |
|---|--:|--:|--:|--:|--:|
| clipart | 68.8 | 68.2 | 69.4 | 77.8 | 77.2 |
| infograph | 34.1 | 33.5 | 34.5 | 42.6 | 43.1 |
| painting | 61.5 | 61.2 | 62.1 | 69.8 | 71.7 |
| quickdraw | 17.3 | 14.9 | 17.7 | 19.1 | 63.3 |
| real | 74.5 | 74.4 | 74.9 | 81.6 | 80.3 |
| sketch | 61.1 | 60.8 | 61.5 | 69.8 | 67.5 |

### 3.2 Adaptation to the held-out domain

Each cell: accuracy at step 0 / median steps to reach 80, 90 and 95% of the ceiling (– = not reached within 500 steps) / AUC ± standard deviation over three 10-shot samples.

| Held-out domain | DINOv3, no DomainNet training (LP-FT) | ERM | KD, source-expert teachers | KD, ViT-B teacher |
|---|--:|--:|--:|--:|
| clipart | 62.4 / 0, –, – / 65.3 ± 0.1 | 68.0 / 0, 30, 150 / 73.5 ± 0.2 | 67.5 / 0, 30, 100 / 73.6 ± 0.2 | 68.7 / 0, 20, 150 / 74.0 ± 0.1 |
| infograph | 23.1 / –, –, – / 26.7 ± 0.8 | 37.6 / 0, 20, 30 / 44.4 ± 0.4 | 37.1 / 0, 20, 30 / 44.6 ± 0.4 | 38.1 / 0, 10, 30 / 44.7 ± 0.5 |
| painting | 52.7 / –, –, – / 56.0 ± 0.3 | 60.1 / 0, 70, – / 66.6 ± 0.1 | 59.9 / 0, 60, – / 66.8 ± 0.1 | 60.6 / 0, 50, 350 / 66.9 ± 0.1 |
| quickdraw | 44.1 / –, –, – / 45.1 ± 0.7 | 17.2 / –, –, – / 43.3 ± 0.3 | 15.0 / –, –, – / 43.2 ± 0.1 | 18.1 / –, –, – / 43.5 ± 0.3 |
| real | 70.5 / 0, 200, – / 72.5 ± 0.6 | 74.3 / 0, 0, 50 / 77.7 ± 0.2 | 74.3 / 0, 0, 50 / 77.6 ± 0.1 | 74.9 / 0, 0, 40 / 78.0 ± 0.2 |
| sketch | 53.3 / 20, –, – / 57.1 ± 0.2 | 63.6 / 0, 0, 20 / 67.4 ± 0.3 | 63.2 / 0, 0, 20 / 67.3 ± 0.3 | 64.0 / 0, 0, 10 / 67.7 ± 0.3 |

Paired AUC difference against ERM (same 10-shot samples), over 6 held-out domains × 3 samples. The 18 comparisons come from 6 trained models per method, so they are not independent.

- DINOv3, no DomainNet training (LP-FT) − ERM: -8.38 (range -18.25 to +2.68; higher in 3 of 18)
- KD, source-expert teachers − ERM: +0.04 (range -0.30 to +0.30; higher in 11 of 18)
- KD, ViT-B teacher − ERM: +0.32 (range +0.18 to +0.55; higher in 18 of 18)

## 4. Domain distance and adaptation

Distances are computed on frozen, pretrained DINOv3 ViT-S features of 20 training images per class: the Fréchet distance between Gaussian fits of the two feature distributions, and OTDD (Alvarez-Melis & Fusi 2020) with a Gaussian approximation of the label-to-label cost.

Pooled five source domains → held-out domain:

| Held-out domain | OTDD | Fréchet |
|---|--:|--:|
| clipart | 5.5 | 4 |
| infograph | 6.2 | 10 |
| painting | 5.9 | 6 |
| quickdraw | 7.0 | 18 |
| real | 6.2 | 7 |
| sketch | 5.3 | 3 |

Spearman correlation between distance and adaptation AUC (a negative ρ means that more distant targets adapt worse). For single-source pairs, p-values come from an exact Mantel test over all 720 relabelings of the six domains; for the six held-out domains, from a permutation test.

| Comparison | Distance | n | Spearman ρ | p |
|---|---|---|---|---|
| Single source, ERM | OTDD | 30 | -0.58 | 0.076 |
| Single source, ERM | Fréchet | 30 | -0.70 | 0.022 |
| Leave-one-out, DINOv3, no DomainNet training (LP-FT) | OTDD | 6 | -0.26 | 0.659 |
| Leave-one-out, ERM | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD, source-expert teachers | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD, ViT-B teacher | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD (ViT-B) − ERM held-out accuracy | OTDD | 6 | 0.03 | 1.000 |

## 5. Limitations

- Each model was trained once. The three repetitions re-sample the 10-shot fine-tuning data only, so differences in held-out accuracy below about one point have no error estimate.
- Infograph and quickdraw have the lowest ceilings and are also the most distant domains, so the distance correlation partly reflects target difficulty.
- With six domains, the leave-one-domain-out analysis has n = 6 per method. Paired comparisons that reuse the same trained model (several targets or 10-shot samples) are not independent, so counts such as "higher in 18 of 18" overstate the evidence if read as independent trials.
- KD and ERM students train for the same number of steps on the same images, but KD additionally requires training the ViT-B teacher, so its total compute is higher.

## Reproducing these results

Requirements beyond requirements.txt: `scipy`, `matplotlib`, `POT` (optimal transport, for OTDD) and `numpy>=2`. DomainNet must be unpacked at `DOMAINNET_ROOT`, as for finetune.py.

```bash
python kd_run_all.py --dry-run   # list the jobs that still need to run
python kd_run_all.py             # run everything (resumable); logs in logs/kd/
python kd_report.py              # rebuild kd_results/RESULTS.md and the figures
```

Single jobs can be run directly, e.g. `python kd_train.py src-student --domain sketch` or `python kd_adapt.py --init models/expert_dinov3_real_s0.pt --target sketch`. Approximate cost on one RTX 3080 Laptop GPU (8 GB): about 25 GPU-hours for the single-source experiments (Section 2) and about 28 GPU-hours for leave-one-domain-out (Section 3).

## Figures

- **fig2_lodo_heldout_accuracy** (`fig2_lodo_heldout_accuracy.png`, `fig2_lodo_heldout_accuracy.svg`): Accuracy on the held-out domain before adaptation. Dashed lines mark the in-domain ceiling. Each model was trained once.
- **fig3_lodo_adaptation_curves** (`fig3_lodo_adaptation_curves.png`, `fig3_lodo_adaptation_curves.svg`): Accuracy on the held-out domain during fine-tuning, one panel per held-out domain. The dashed line marks 90% of the in-domain ceiling.
- **fig4_distance_vs_adaptation** (`fig4_distance_vs_adaptation.png`, `fig4_distance_vs_adaptation.svg`): Adaptation AUC against OTDD. Left: one point per (source, target) pair. Right: one point per held-out domain and starting model; points are offset horizontally by up to 0.05 so that overlapping methods stay visible.

_Generated by kd_report.py from results/*.json._
