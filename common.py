import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import random
import warnings
from collections import defaultdict
from pathlib import Path

import timm
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

warnings.filterwarnings("ignore", message=".*still experimental.*")

ROOT = Path(os.environ.get("DOMAINNET_ROOT", "~/stuff/data/domainnet")).expanduser()
DOMAINS = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
NUM_CLASSES = 345
BACKBONES = {
    "dinov3": "vit_small_patch16_dinov3.lvd1689m",
    "dino": "vit_small_patch16_224.dino",
    "deit": "deit_small_patch16_224.fb_in1k",
    "resnet": "resnet50.tv_in1k",
}

BICUBIC = T.InterpolationMode.BICUBIC
NORMALIZE = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, (0.5, 1), interpolation=BICUBIC), T.RandomHorizontalFlip(), T.ToTensor(), NORMALIZE])
TEST_TF = T.Compose([T.Resize(256, BICUBIC), T.CenterCrop(224), T.ToTensor(), NORMALIZE])


def backbone(name, num_classes=0, pretrained=True):
    return timm.create_model(BACKBONES[name], pretrained=pretrained, num_classes=num_classes).cuda()


class DomainNet(Dataset):
    def __init__(self, domain, split, train=False, per_class=0, limit=0, seed=0):
        lines = (ROOT / f"{domain}_{split}.txt").read_text().split()
        items = [(path, int(label)) for path, label in zip(lines[::2], lines[1::2])]
        if per_class:
            by_class = defaultdict(list)
            for item in items:
                by_class[item[1]].append(item)
            rng = random.Random(seed)
            items = [x for c in sorted(by_class) for x in (by_class[c] if len(by_class[c]) <= per_class else rng.sample(by_class[c], per_class))]
        if limit:
            items = random.Random(seed).sample(items, limit)
        self.items, self.tf = items, TRAIN_TF if train else TEST_TF

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return self.tf(Image.open(ROOT / path).convert("RGB")), label


def test_loader(domain, limit=0, seed=0):
    return DataLoader(DomainNet(domain, "test", limit=limit, seed=seed), batch_size=256, num_workers=10)


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    hits = 0
    for x, y in loader:
        with torch.autocast("cuda", torch.bfloat16):
            hits += (model(x.cuda()).argmax(1).cpu() == y).sum().item()
    model.train()
    return hits / len(loader.dataset)
