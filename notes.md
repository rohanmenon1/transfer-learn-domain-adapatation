# Notes

## Files

- `common.py`: shared pieces: DomainNet dataset loader, backbone registry (`resnet`, `deit`, `dino`, `dinov3` via timm), image transforms and an accuracy helper.
- `probe.py`: freezes a pretrained backbone, extracts its features once in memory, trains a linear classifier on each source domain and tests it on every domain. Writes `results/probe_<backbone>_pc<n>.json`.
- `finetune.py`: fine-tunes the whole backbone plus classifier on one source domain, logging accuracy on every domain after each epoch. Writes `results/ft_*.json` and saves the model to `models/`.
- `evaluate.py`: loads a saved model and adds its accuracy on the full test sets to the run's results file. Runs as a separate process because long GPU processes sometimes crash on our AMD card.

Data is expected at `~/stuff/data/domainnet` (override with `DOMAINNET_ROOT`).

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
