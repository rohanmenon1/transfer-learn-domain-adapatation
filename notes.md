# Notes

## Files

- `common.py`: shared pieces: DomainNet dataset loader (reads the Hugging Face copy `wltjr1007/DomainNet`, downloaded on first use), backbone registry (`resnet`, `deit`, `dino`, `dinov3` via timm), image transforms and an accuracy helper.
- `probe.py`: freezes a pretrained backbone, extracts its features once in memory, trains a linear classifier on each source domain and tests it on every domain. Writes `results/probe_<backbone>_pc<n>.json`.
- `finetune.py`: fine-tunes the whole backbone plus classifier on one source domain, logging accuracy on every domain after each epoch. Writes `results/ft_*.json` and saves the model to `models/`.
- `evaluate.py`: loads a saved model and adds its accuracy on the full test sets to the run's results file. Runs as a separate process because long GPU processes sometimes crash on our AMD card.
- `train_from_scratch.py`: trains a ResNet (`--arch resnet18|resnet50`) on a domain's full train set, from random weights or with `--pretrained` from ImageNet weights, then evaluates on every domain's full test set. Writes `results/{scratch,pre}_<arch>_<domain>.json`.
- `compare.py`: prints a final-accuracy table for any result files passed to it.
- `run.sh`: runs a command with retries, restarting it if its log goes quiet for 25 min (GPU freeze). Training resumes from the last finished epoch.
- `experiments.sh`: the full-data experiment queue (ResNet-50 from scratch and pretrained, DINOv3 fine-tuning).

Data comes from Hugging Face (`hf download wltjr1007/DomainNet --repo-type dataset`, 18.5 GB; it matches the official cleaned train/test split).

## Run

```bash
python probe.py --backbone dinov3
python finetune.py --backbone dinov3 --source real
python evaluate.py ft_dinov3_real_pc50_s0
```

## Current results

Training uses at most 50 images per class (~16k per domain, 96.6k total = 24% of the 409.8k full train set). Testing uses the full official test sets (176.7k images, 14.6k–52.0k per domain). Fine-tuning runs 10 epochs. Accuracy is top-1 in %.

**Reading a cell.** Each cell is `A / B`. Rows are the training domain and columns the test domain.

- **A**: the pretrained weights are kept frozen, and only a linear classifier is trained on the row's domain, then evaluated on the column's domain.
- **B**: same, but the pretrained weights are also fine-tuned on the row's domain.

Example: ResNet-50, row clipart, column sketch = `20.8 / 35.7`. Trained on clipart and tested on sketch, it gets 20.8% with frozen weights and 35.7% after fine-tuning. Diagonal cells (same train and test domain) are bold. Diagonal cells tell us about how much gain the model gets from fine tuning, essentially telling us how much space there is for improvement (with this amount of data).

## ResNet-50

23.5M params · Dec 2015 · pretrained supervised on ImageNet-1k (1.3M labeled images)

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **51.9 / 68.4** | 10.1 / 14.9 | 24.5 / 31.3 | 2.2 / 8.0 | 43.1 / 49.9 | 20.8 / 35.7 |
| infograph | 19.0 / 28.5 | **21.4 / 31.5** | 22.8 / 29.6 | 0.8 / 1.7 | 39.3 / 47.0 | 13.7 / 21.4 |
| painting | 24.0 / 34.0 | 10.3 / 13.9 | **51.2 / 61.2** | 1.0 / 3.0 | 51.5 / 56.0 | 17.9 / 28.5 |
| quickdraw | 1.6 / 3.3 | 0.7 / 0.5 | 1.1 / 0.6 | **38.8 / 58.9** | 1.6 / 2.1 | 1.9 / 4.5 |
| real | 25.9 / 36.1 | 11.4 / 16.1 | 34.7 / 39.9 | 0.9 / 3.0 | **71.6 / 75.2** | 17.8 / 24.8 |
| sketch | 26.8 / 40.4 | 9.1 / 12.1 | 24.5 / 29.2 | 2.5 / 9.8 | 37.9 / 44.3 | **38.4 / 58.2** |

Mean in-domain 45.6 → 58.9; mean cross-domain 16.6 → 22.3.

## DeiT-S

21.7M params · Dec 2020 · pretrained supervised on ImageNet-1k (1.3M labeled images)

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **57.2 / 72.1** | 14.4 / 20.0 | 30.8 / 42.0 | 3.1 / 10.1 | 48.1 / 58.7 | 25.8 / 41.1 |
| infograph | 23.7 / 37.2 | **26.0 / 35.7** | 29.8 / 40.6 | 1.8 / 4.5 | 46.3 / 58.1 | 18.0 / 30.1 |
| painting | 28.7 / 41.8 | 13.8 / 19.1 | **54.5 / 65.1** | 1.0 / 3.7 | 57.8 / 62.2 | 21.8 / 32.8 |
| quickdraw | 3.0 / 17.7 | 0.8 / 4.3 | 0.6 / 8.3 | **44.3 / 59.8** | 1.3 / 13.3 | 2.6 / 12.4 |
| real | 29.7 / 40.8 | 13.8 / 19.2 | 40.9 / 46.0 | 1.2 / 3.5 | **73.9 / 77.4** | 21.2 / 29.5 |
| sketch | 32.6 / 49.7 | 12.9 / 19.2 | 30.8 / 41.2 | 3.5 / 12.5 | 45.8 / 57.6 | **43.1 / 61.1** |

Mean in-domain 49.9 → 61.9; mean cross-domain 20.2 → 29.3.

## DINO v1 ViT-S/16

21.7M params · Apr 2021 · pretrained self-supervised on ImageNet-1k (no labels)

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **58.4 / 69.8** | 13.1 / 15.5 | 30.1 / 34.5 | 3.0 / 9.1 | 44.8 / 49.4 | 25.0 / 37.0 |
| infograph | 20.4 / 27.5 | **27.7 / 32.2** | 30.2 / 30.9 | 1.0 / 1.9 | 45.0 / 46.7 | 16.2 / 21.5 |
| painting | 23.5 / 34.6 | 12.9 / 15.8 | **58.4 / 62.7** | 0.5 / 2.6 | 53.3 / 55.0 | 19.2 / 29.1 |
| quickdraw | 2.6 / 15.5 | 1.1 / 2.1 | 1.4 / 5.9 | **43.4 / 60.2** | 1.8 / 10.4 | 3.0 / 10.6 |
| real | 25.1 / 35.5 | 13.4 / 17.4 | 39.9 / 41.9 | 0.9 / 2.4 | **74.0 / 75.5** | 18.7 / 25.5 |
| sketch | 30.6 / 44.4 | 12.7 / 15.9 | 33.2 / 35.5 | 3.1 / 12.5 | 43.5 / 48.4 | **45.0 / 58.8** |

Mean in-domain 51.1 → 59.8; mean cross-domain 19.0 → 24.5.

## DINOv3 ViT-S/16

21.6M params · Aug 2025 · pretrained self-supervised on LVD-1689M (1.7B web images, no labels), distilled from a 7B teacher

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **71.2 / 77.2** | 16.1 / 26.9 | 38.4 / 50.5 | 10.4 / 13.1 | 54.4 / 63.0 | 44.3 / 53.9 |
| infograph | 38.2 / 47.7 | **33.5 / 43.1** | 36.5 / 48.5 | 6.3 / 8.7 | 52.4 / 61.6 | 35.6 / 45.5 |
| painting | 41.2 / 52.6 | 14.1 / 24.5 | **63.9 / 71.7** | 3.6 / 6.6 | 57.6 / 63.8 | 38.0 / 47.2 |
| quickdraw | 14.5 / 29.9 | 2.4 / 9.4 | 6.4 / 18.7 | **55.0 / 63.3** | 9.1 / 24.9 | 11.9 / 25.4 |
| real | 41.5 / 52.3 | 14.4 / 27.0 | 45.1 / 54.1 | 4.9 / 7.0 | **77.4 / 80.3** | 36.1 / 45.5 |
| sketch | 52.4 / 61.3 | 16.0 / 27.5 | 40.2 / 50.7 | 11.6 / 15.8 | 52.6 / 63.9 | **60.8 / 67.5** |

Mean in-domain 60.3 → 67.2; mean cross-domain 28.2 → 37.6.

## Full data (all train images, full test sets)

### ResNet-50 from scratch, 224px, 40 epochs

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **54.2** | 5.4 | 10.1 | 5.9 | 20.0 | 14.4 |
| infograph | 7.6 | **20.3** | 6.5 | 0.7 | 10.9 | 4.4 |
| painting | 16.0 | 5.6 | **46.5** | 1.0 | 27.9 | 12.2 |
| quickdraw | 3.4 | 0.5 | 0.2 | **71.1** | 0.9 | 3.4 |
| real | 30.3 | 10.3 | 27.7 | 3.4 | **70.0** | 16.3 |
| sketch | 24.4 | 5.6 | 13.4 | 8.6 | 17.8 | **52.3** |

### ResNet-50 ImageNet-pretrained, 224px, 10 epochs, lr 1e-4

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **75.0** | 15.7 | 30.4 | 10.2 | 45.4 | 35.8 |
| infograph | 26.2 | **39.2** | 25.1 | 2.2 | 38.0 | 21.1 |
| painting | 34.5 | 16.1 | **69.3** | 3.2 | 50.7 | 29.9 |
| quickdraw | 6.8 | 0.7 | 1.1 | **72.4** | 3.1 | 8.5 |
| real | 44.2 | 20.3 | 44.1 | 5.2 | **82.2** | 32.1 |
| sketch | 44.0 | 12.3 | 29.4 | 11.0 | 38.7 | **67.2** |

### DINOv3 ViT-S/16 fine-tuned, 10 epochs

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **81.4** | 26.8 | 49.7 | 14.2 | 62.6 | 54.2 |
| infograph | 48.4 | **49.2** | 47.0 | 7.1 | 60.6 | 43.8 |
| painting | 54.0 | 26.7 | **77.6** | 5.9 | 64.5 | 47.7 |
| quickdraw | 26.0 | 5.0 | 12.3 | **73.5** | 18.2 | 23.6 |
| real | 58.8 | 28.7 | 56.8 | 8.5 | **86.6** | 49.4 |
| sketch | 60.6 | 23.1 | 45.2 | 17.0 | 59.2 | **74.4** |

## OTDD

`distance.py`: 100 train images/class, PCA 64, exact OT, 3 seeds of disjoint halves (source half 0, target half 1). Diagonal is the self-distance floor. Std over seeds is at most 0.07, so it is omitted. The second table is distance minus sqrt(floor_a * floor_b).

### resnet features

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **15.82** | 21.96 | 21.53 | 27.66 | 21.73 | 19.67 |
| infograph | 21.97 | **14.87** | 24.08 | 30.24 | 24.31 | 22.92 |
| painting | 21.51 | 24.03 | **16.30** | 29.99 | 19.19 | 21.49 |
| quickdraw | 27.69 | 30.19 | 30.01 | **8.97** | 30.13 | 25.24 |
| real | 21.70 | 24.22 | 19.20 | 30.12 | **14.79** | 22.25 |
| sketch | 19.70 | 22.88 | 21.52 | 25.33 | 22.23 | **15.03** |

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **0.00** | 6.62 | 5.47 | 15.75 | 6.44 | 4.25 |
| infograph | 6.64 | **0.00** | 8.51 | 18.69 | 9.48 | 7.97 |
| painting | 5.45 | 8.46 | **0.00** | 17.90 | 3.66 | 5.84 |
| quickdraw | 15.78 | 18.65 | 17.91 | **0.00** | 18.61 | 13.63 |
| real | 6.40 | 9.39 | 3.67 | 18.60 | **0.00** | 7.34 |
| sketch | 4.28 | 7.94 | 5.86 | 13.71 | 7.32 | **0.00** |

### dinov3 features

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **3.56** | 5.61 | 5.28 | 5.94 | 5.23 | 4.40 |
| infograph | 5.61 | **3.41** | 5.94 | 7.17 | 5.97 | 5.56 |
| painting | 5.30 | 5.93 | **3.58** | 6.99 | 4.84 | 4.90 |
| quickdraw | 5.94 | 7.17 | 7.00 | **2.67** | 7.17 | 5.69 |
| real | 5.23 | 5.96 | 4.83 | 7.16 | **3.45** | 5.09 |
| sketch | 4.42 | 5.55 | 4.90 | 5.71 | 5.08 | **3.50** |

| train \ test | clipart | infograph | painting | quickdraw | real | sketch |
|---|---|---|---|---|---|---|
| clipart | **0.00** | 2.12 | 1.72 | 2.86 | 1.73 | 0.87 |
| infograph | 2.13 | **0.00** | 2.44 | 4.16 | 2.55 | 2.10 |
| painting | 1.73 | 2.44 | **0.00** | 3.89 | 1.32 | 1.36 |
| quickdraw | 2.86 | 4.15 | 3.91 | **0.00** | 4.14 | 2.63 |
| real | 1.72 | 2.53 | 1.32 | 4.12 | **0.00** | 1.61 |
| sketch | 0.89 | 2.10 | 1.36 | 2.66 | 1.60 | **0.00** |

### Distance vs transfer (Spearman rho, off-diagonal pairs; distance = minus floor)

| distance | accuracy of | all 30 acc | all 30 rel | no quickdraw (20) acc | no quickdraw rel |
|---|---|---|---|---|---|
| resnet | R50 scratch | -0.84 | -0.78 | -0.61 | -0.39 |
| resnet | R50 pretrained | -0.87 | -0.73 | -0.58 | -0.13 |
| resnet | DINOv3 | -0.82 | -0.67 | -0.47 | -0.01 |
| dinov3 | R50 scratch | -0.85 | -0.76 | -0.65 | -0.30 |
| dinov3 | R50 pretrained | -0.86 | -0.71 | -0.56 | -0.06 |
| dinov3 | DINOv3 | -0.81 | -0.65 | -0.43 | 0.06 |

rel = target accuracy / source in-domain accuracy. resnet and dinov3 distance rankings agree (rho 0.96, 0.86 without quickdraw).
