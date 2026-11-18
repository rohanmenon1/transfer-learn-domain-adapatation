# Knowledge distillation and ERM on DomainNet

Adaptation speed of DINOv3 ViT-S models trained with ERM or knowledge distillation (KD), measured as accuracy on a new domain during few-shot fine-tuning.

## Summary

- **Train on two domains (KD, n = 60 runs):** source pairs farther from the target take longer to converge. Within target, ρ = +0.46 (p = 0.001) for the plateau step and +0.72 (p = 0.007) for steps to 95% of final accuracy; pooled over targets, ρ = +0.54 (p = 0.149) and +0.77 (p = 0.028). On quickdraw, every source pair converges late.
- **Single-source transfer:** KD with a ViT-B teacher differs from ERM in adaptation AUC by +0.11 points on average (higher in 68 of 90 comparisons).
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
| Training length | one or two source domains: 10 epochs over the source images (finetune.py); five source domains: 5,000 steps (DomainBed default); all six domains (reference): 10 epochs |
| ERM | cross-entropy on the source data |
| KD | Hinton et al. (2015): (1 − w) cross-entropy + w KL divergence to the teacher at temperature T. Sections 2.1-2.2 and 3: T = 1, w = 0.5; Section 5: T = 4, w = 0.5, chosen by the sweep in Section 2.3 |
| Baseline | pretrained DINOv3 without DomainNet training. Its classifier head is first fitted on the 10-shot data with the backbone frozen, then the whole model is fine-tuned (LP-FT, Kumar et al. 2022); a randomly initialised head would put it at an unfair disadvantage. ERM and KD models keep their trained head. |
| Adaptation test | fine-tune on 10 labelled images per class of the target domain for 500 steps; accuracy on a fixed class-balanced subset of the target test set (20 images per class) |
| Adaptation metrics | steps to reach 80%, 90% and 95% of the target's in-domain ceiling; AUC = mean accuracy over the 500 steps |
| Ceiling | in-domain accuracy of DINOv3 ViT-S fine-tuned on the target domain (notes.md) |
| Repetitions | each model is trained once; adaptation tests in Sections 2.2 and 3 are repeated with 3 different 10-shot samples, those in Sections 2.3 and 5 use one sample |

As a pipeline check, the single-domain ERM model for sketch reaches 67.6% on the sketch test set (finetune.py: 67.5%).

## 2. Single-source transfer (train on one domain, adapt to the other five)

Each model is trained on one source domain and then adapted to each of the five other domains. The ERM model is the single-domain model trained with finetune.py's recipe; the KD model is trained on the same images for the same number of steps, with a ViT-B teacher trained on that domain.

### 2.1 Accuracy before adaptation

Top-1 accuracy (%) on the source domain's full test set, and the mean over the other five domains (2,000 test images each), before any target data is used.

| Source | ERM, source | ERM, other five | KD, source | KD, other five | ViT-B teacher, source | ViT-B teacher, other five |
|---|--:|--:|--:|--:|--:|--:|
| clipart | 77.4 | 41.8 | 77.8 | 41.6 | 83.4 | 50.0 |
| infograph | 43.5 | 42.3 | 43.9 | 42.9 | 51.7 | 54.0 |
| painting | 71.7 | 38.8 | 72.1 | 39.0 | 78.1 | 48.1 |
| quickdraw | 63.4 | 22.9 | 64.1 | 22.8 | 67.3 | 35.1 |
| real | 80.5 | 37.7 | 80.5 | 37.2 | 85.2 | 46.6 |
| sketch | 67.6 | 45.0 | 67.9 | 44.9 | 75.2 | 51.9 |

### 2.2 Adaptation to each target

Mean AUC over the five targets and three 10-shot samples, and the paired difference KD − ERM (same 10-shot samples for both models).

| Source | ERM, mean AUC | KD, mean AUC | KD − ERM, AUC points |
|---|--:|--:|--:|
| clipart | 56.3 | 56.3 | +0.06 (range -0.26 to +0.27; higher in 11 of 15) |
| infograph | 59.0 | 59.2 | +0.23 (range -0.03 to +0.45; higher in 14 of 15) |
| painting | 55.8 | 55.9 | +0.17 (range -0.08 to +0.36; higher in 13 of 15) |
| quickdraw | 49.8 | 49.9 | +0.07 (range -0.17 to +0.37; higher in 10 of 15) |
| real | 53.7 | 53.7 | +0.07 (range -0.12 to +0.39; higher in 9 of 15) |
| sketch | 57.8 | 57.8 | +0.04 (range -0.29 to +0.25; higher in 11 of 15) |

**All pairs:** KD − ERM = +0.11 (range -0.29 to +0.45; higher in 68 of 90) over 90 of 90 (source, target, sample) comparisons. Comparisons that share a model are not independent.

### 2.3 KD hyperparameters

Temperature T and KD weight w (cross-entropy weight 1 − w) for single-domain students trained on clipart and sketch and adapted to the other five domains. Each cell is the mean paired AUC difference against the ERM model of the same source over the five targets, using the first 10-shot sample only.

| Setting | KD − ERM AUC, source clipart | KD − ERM AUC, source sketch |
|---|---|---|
| T = 1, w = 0.5 (setting of Section 2.2) | +0.00 (higher in 2 of 5) | +0.07 (higher in 4 of 5) |
| T = 2, w = 0.5 | -0.12 (higher in 1 of 5) | -0.08 (higher in 1 of 5) |
| T = 4, w = 0.5 | +0.36 (higher in 5 of 5) | +0.36 (higher in 5 of 5) |
| T = 2, w = 0.9 | -0.13 (higher in 1 of 5) | +0.01 (higher in 3 of 5) |
| T = 4, w = 0.9 | +0.20 (higher in 4 of 5) | +0.23 (higher in 4 of 5) |

T = 4, w = 0.5 gives the largest difference and is used for the two-source students in Section 5. With two sources and one 10-shot sample, the sweep selects a setting rather than establishing an effect.

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
| Single source, KD, ViT-B teacher | OTDD | 30 | -0.59 | 0.064 |
| Single source, KD, ViT-B teacher | Fréchet | 30 | -0.71 | 0.019 |
| Leave-one-out, DINOv3, no DomainNet training (LP-FT) | OTDD | 6 | -0.26 | 0.659 |
| Leave-one-out, ERM | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD, source-expert teachers | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD, ViT-B teacher | OTDD | 6 | -0.37 | 0.495 |
| Leave-one-out, KD (ViT-B) − ERM held-out accuracy | OTDD | 6 | 0.03 | 1.000 |

## 5. Train on two domains, adapt to each of the other four

A KD student is trained on each of the 15 pairs of source domains (10 epochs over both domains' images); each image is distilled from the single-domain ViT-B teacher of its own domain (T = 4, w = 0.5). Each student is then adapted to each of the four other domains with the adaptation test of Section 1 (one 10-shot sample), giving 60 (source pair, target) runs.

- **Convergence:** the plateau step is the first evaluation step after which accuracy stays within 1.5 points of the run's final accuracy, reported as a percentage of the 500 fine-tuning steps. Steps to 95% of the run's final accuracy and AUC are reported as well. Accuracy is evaluated every 10 steps up to step 100 and every 50 steps after that, so step-based values are coarse.
- **Distance:** shared OTDD table (DINOv3 features of 100 training images per class, noise floor subtracted); the distance of a source pair to a target is the mean of the two sources' distances.
- **Tests:** Spearman ρ, pooled over all runs and within target (mean of the six per-target ρ, which removes differences in target difficulty); p-values from all 720 relabelings of the six domains.

| Convergence measure | n | Pooled ρ | p | Within-target ρ | p |
|---|---|---|---|---|---|
| % of training at plateau | 60 | +0.54 | 0.149 | +0.46 | 0.001 |
| steps to 95% of final | 60 | +0.77 | 0.028 | +0.72 | 0.007 |
| AUC (higher = better) | 60 | -0.65 | 0.065 | -0.82 | 0.006 |

A positive ρ for the step measures means that source pairs farther from the target take longer; a negative ρ for AUC means that they adapt worse.

With the OTDD of Section 4 in place of the shared table, the within-target ρ is +0.43 (p = 0.014) for the plateau and +0.70 (p = 0.007) for steps to 95% of final accuracy.

Per target:

| Target | n | Plateau, % of training (mean) | Steps to 95% of final (mean) | Final accuracy (mean) | ρ, distance vs plateau |
|---|--:|--:|--:|--:|--:|
| clipart | 10 | 34 | 88 | 72.3 | +0.36 |
| infograph | 10 | 27 | 89 | 43.2 | +0.47 |
| painting | 10 | 38 | 103 | 65.8 | +0.62 |
| quickdraw | 10 | 63 | 280 | 47.8 | -0.11 |
| real | 10 | 26 | 61 | 76.6 | +0.58 |
| sketch | 10 | 32 | 82 | 66.4 | +0.83 |

`pairs_kd.csv` has one row per run: source pair, target, distance, plateau step and percentage, steps to 95% of final accuracy, AUC, and accuracy at the first and last step.

## 6. Reference: ERM on all six domains

Upper reference for accuracy: a DINOv3 ViT-S trained with ERM on all six domains (50 images per class and domain, 10 epochs), including each target. It is compared with the final accuracy of the two-source KD students after adaptation, which see 10 images per class of the target.

| Domain | Reference, full test set | Reference, adaptation test subset | Ceiling (one domain) | Two-source KD after adaptation, mean (range) | Difference to reference |
|---|---|---|---|---|---|
| clipart | 77.7 | 77.7 | 77.2 | 72.3 (71.0–73.8) | -5.3 |
| infograph | 44.6 | 48.7 | 43.1 | 43.2 (41.5–44.8) | -5.4 |
| painting | 71.7 | 70.8 | 71.7 | 65.8 (62.9–67.6) | -5.0 |
| quickdraw | 62.1 | 61.6 | 63.3 | 47.8 (45.6–49.9) | -13.8 |
| real | 81.2 | 80.8 | 80.3 | 76.6 (74.8–78.0) | -4.2 |
| sketch | 68.7 | 72.1 | 67.5 | 66.4 (64.3–68.0) | -5.6 |

Accuracy in %. The adaptation test subset (20 images per class) is the one used for every adaptation run, so it can be compared directly with the last two columns. The ceiling is the single-domain model of notes.md, measured on the full test set.

## 7. Limitations

- Each model was trained once. The three repetitions re-sample the 10-shot fine-tuning data only, so differences in held-out accuracy below about one point have no error estimate.
- Infograph and quickdraw have the lowest ceilings and are also the most distant domains, so the distance correlation partly reflects target difficulty.
- With six domains, the leave-one-domain-out analysis has n = 6 per method. Paired comparisons that reuse the same trained model (several targets or 10-shot samples) are not independent, so counts such as "higher in 18 of 18" overstate the evidence if read as independent trials.
- KD and ERM students train for the same number of steps on the same images, but KD additionally requires training the ViT-B teacher, so its total compute is higher.
- Section 5 has KD students only, one 10-shot sample per run, and no ERM model trained on the same source pairs, so it describes how distance relates to convergence for KD but does not compare methods.

## Reproducing these results

Requirements beyond requirements.txt: `scipy`, `matplotlib`, `POT` (optimal transport, for OTDD) and `numpy>=2`. DomainNet must be unpacked at `DOMAINNET_ROOT`, as for finetune.py.

```bash
python kd_run_all.py --dry-run   # list the jobs that still need to run
python kd_run_all.py             # run everything (resumable); logs in logs/kd/
python kd_report.py              # rebuild kd_results/RESULTS.md and the figures
```

Single jobs can be run directly, e.g. `python kd_train.py src-student --domain sketch` or `python kd_adapt.py --init models/expert_dinov3_real_s0.pt --target sketch`. Approximate cost on one RTX 3080 Laptop GPU (8 GB): about 25 GPU-hours for the single-source experiments (Sections 2.1-2.2), about 5 GPU-hours for the KD hyperparameter sweep (Section 2.3), about 28 GPU-hours for leave-one-domain-out (Section 3), about 14 GPU-hours for the two-source experiments (Section 5) and about 2 GPU-hours for the reference model (Section 6).

## Figures

- **fig1_single_source_kd_vs_erm** (`fig1_single_source_kd_vs_erm.png`, `fig1_single_source_kd_vs_erm.svg`): Paired difference in adaptation AUC between the KD and ERM models for every (source, target) pair, averaged over three 10-shot samples.
- **fig2_lodo_heldout_accuracy** (`fig2_lodo_heldout_accuracy.png`, `fig2_lodo_heldout_accuracy.svg`): Accuracy on the held-out domain before adaptation. Dashed lines mark the in-domain ceiling. Each model was trained once.
- **fig3_lodo_adaptation_curves** (`fig3_lodo_adaptation_curves.png`, `fig3_lodo_adaptation_curves.svg`): Accuracy on the held-out domain during fine-tuning, one panel per held-out domain. The dashed line marks 90% of the in-domain ceiling.
- **fig4_distance_vs_adaptation** (`fig4_distance_vs_adaptation.png`, `fig4_distance_vs_adaptation.svg`): Adaptation AUC against OTDD. Left: one point per (source, target) pair. Right: one point per held-out domain and starting model; points are offset horizontally by up to 0.05 so that overlapping methods stay visible.
- **fig5_pairs_distance_vs_plateau** (`fig5_pairs_distance_vs_plateau.png`, `fig5_pairs_distance_vs_plateau.svg`): One point per (source pair, target) run: distance against the share of fine-tuning completed when accuracy settles within 1.5 points of its final value.
- **fig6_pairs_by_target** (`fig6_pairs_by_target.png`, `fig6_pairs_by_target.svg`): The runs of fig5 split by target domain; each panel compares the ten source pairs on the same target, so target difficulty is held fixed.

_Generated by kd_report.py from results/*.json._
