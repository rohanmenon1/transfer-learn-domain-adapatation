# DINOv3 model, image transforms, validation, and checkpoint utilities.
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import timm

DOMAINS = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
CLASSES = 345


# Seed Python, NumPy, and PyTorch for reproducible initialization.
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Read and validate the supplied domain split and numeric labels.
def read_split(root, domain, split):
    path = root / 'splits' / f'{domain}_{split}.txt'
    rows = []
    for line in path.read_text().splitlines():
        name, label = line.rsplit(maxsplit=1)
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != domain:
            raise ValueError(f'Unsafe or wrong-domain path: {name}')
        label = int(label)
        if not 0 <= label < CLASSES:
            raise ValueError(f'Invalid label: {label}')
        rows.append((name, label))
    if not rows or len({x[0] for x in rows}) != len(rows):
        raise ValueError(f'Empty split or duplicate image paths: {path}')
    return rows


# Hold out stratified validation samples from the training split.
def partition(rows, fraction, seed):
    """Stratify validation using only the official training split."""
    rng = random.Random(seed)
    train, val = [], []
    groups = {}
    for row in rows:
        groups.setdefault(row[1], []).append(row)
    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        n = min(len(group) - 1, max(1, round(len(group) * fraction)))
        val.extend(group[:n])
        train.extend(group[n:])
    if not val:
        raise ValueError('No validation samples; need multiple examples per class')
    return train, val


class Images(Dataset):
    # Configure training augmentation or deterministic evaluation transforms.
    def __init__(self, root, rows, training=False):
        self.root, self.rows = root, rows
        ops = [T.RandomResizedCrop(256, interpolation=T.InterpolationMode.BICUBIC), T.RandomHorizontalFlip()] if training else [T.Resize(256, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(256)]
        self.transform = T.Compose(ops + [T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])

    # Return the number of images in this split.
    def __len__(self):
        return len(self.rows)

    # Load one RGB image, apply transforms, and return its label.
    def __getitem__(self, index):
        name, label = self.rows[index]
        with Image.open(self.root / name) as im:
            return self.transform(im.convert('RGB')), label


# Build the image loader with the configured batch size and workers.
def loader(args, rows, training=False):
    return DataLoader(Images(args.data, rows, training), batch_size=args.batch_size,
                      shuffle=training, num_workers=args.workers,
                      pin_memory=args.device.startswith('cuda'),
                      generator=torch.Generator().manual_seed(args.seed))


# Copy model tensors to CPU for portable checkpoints.
def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# Write model weights and metadata through a temporary file.
def save_checkpoint(path, state, metadata):
    tmp = path.with_suffix('.tmp')
    torch.save({'state': state, 'metadata': metadata}, tmp)
    tmp.replace(path)


# Load a local tensor checkpoint using restricted deserialization.
def load_checkpoint(path):
    return torch.load(path, map_location='cpu', weights_only=True)


# Compute top-1 accuracy, macro accuracy, and loss without gradients.
@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    hits = torch.zeros(CLASSES, dtype=torch.long)
    counts = torch.zeros_like(hits)
    loss_sum = 0.
    for x, y in batches:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=device.startswith('cuda')):
            logits = model(x.to(device))
        logits = logits.float()
        loss_sum += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        pred = logits.argmax(1).cpu()
        counts += torch.bincount(y, minlength=CLASSES)
        hits += torch.bincount(y[pred == y], minlength=CLASSES)
    return {'accuracy': hits.sum().item() / counts.sum().item(),
            'macro_accuracy': (hits[counts > 0] / counts[counts > 0]).float().mean().item(),
            'loss': loss_sum / counts.sum().item(), 'n': counts.sum().item()}



# Create DINOv3 ViT-S/16 with a shared 345-class classifier layout.
def make_model():
    backbone = timm.create_model('vit_small_patch16_dinov3', pretrained=False,
                                 num_classes=0, global_pool='avg')
    return nn.Sequential(backbone, nn.Linear(backbone.num_features, CLASSES))


# Replace a JSON file after its temporary file has been written.
def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)
