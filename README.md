# DomainNet DINOv3 weight averaging

Six source models, fifteen pair averages, and sixty independent target adaptations across the six DomainNet domains.

## Files

- `experiments/dinov3/`: exact Python snapshot saved by the run.
- `results/dinov3/`: configuration, shared image manifest, and metrics.
- `checkpoints/dinov3/`: source finals, pair averages, and target finals (Git LFS).
- `publication.json`: experiment and publication provenance.

Training uses up to 50 images per class from the official train split. Monitoring uses 2,000 fixed official-test images per target. Source models train for a fixed 10 epochs. Target adaptation checks a two-epoch, 0.5-percentage-point plateau every 100 steps, with a 10-epoch cap. Convergence is the first observed step reaching 95% of the run's best observed accuracy. The monitoring subset is used for adaptive stopping, so it is not an untouched final test.

Optimizer-resume checkpoints remain in Drive and are not published. The previous branch contents remain available in Git history.
