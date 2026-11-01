import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
import random
from collections import defaultdict
from pathlib import Path

HF_DATASET = "wltjr1007/DomainNet"


import torch
from datasets import load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import warnings
import timm

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


def load_domainnet_from_hub():
    """Download/cache the Hugging Face dataset and inspect its schema."""
    ds = load_dataset(HF_DATASET)

    print("Dataset splits:", list(ds.keys()))
    for split, data in ds.items():
        print(f"{split}: {len(data):,} examples")
        print("Columns:", data.column_names)
        print("Features:", data.features)

    return ds


   

class DomainNet(Dataset):
    _dataset = None

    def __init__(
        self,
        domain,
        split,
        train=False,
        per_class=0,
        limit=0,
        seed=0,
    ):
        # Download/cache the dataset once per Python process.
        if DomainNet._dataset is None:
            DomainNet._dataset = load_dataset(HF_DATASET)

        ds = DomainNet._dataset

        if split not in ds:
            raise ValueError(
                f"Unknown split {split!r}. "
                f"Available splits: {list(ds.keys())}"
            )

        data = ds[split]

        # Inspect the schema to identify the actual column names.
        columns = data.column_names

        def find_column(candidates):
            return next(
                (c for c in candidates if c in columns),
                None,
            )

        image_col = find_column(["image", "img"])
        label_col = find_column(["label", "labels", "category"])
        domain_col = find_column(
            ["domain", "domain_name", "domain_id"]
        )

        if not all([image_col, label_col, domain_col]):
            raise ValueError(
                f"Cannot identify image, label, and domain columns. "
                f"Columns: {columns}. "
                f"Inspect the dataset schema to update the candidates."
            )

        domain_feature = data.features[domain_col]
        label_feature = data.features[label_col]

        def feature_value(feature, value):
            if hasattr(feature, "int2str") and isinstance(value, int):
                return feature.int2str(value)
            return str(value)

        # Select rows belonging to the requested domain.
        domain_indices = []
        for i, value in enumerate(data[domain_col]):
            if feature_value(domain_feature, value).lower() == domain.lower():
                domain_indices.append(i)

        if not domain_indices:
            raise ValueError(
                f"No rows found for domain {domain!r}. "
                f"Example domain values: {data[domain_col][:10]}"
            )

        items = [
            (i, int(data[i][label_col]))
            for i in domain_indices
        ]

        # Optional class-balanced sampling.
        if per_class:
            by_class = defaultdict(list)
            for item in items:
                by_class[item[1]].append(item)

            rng = random.Random(seed)
            items = [
                item
                for label in sorted(by_class)
                for item in (
                    by_class[label]
                    if len(by_class[label]) <= per_class
                    else rng.sample(by_class[label], per_class)
                )
            ]

        if limit:
            if limit > len(items):
                raise ValueError(
                    f"Requested {limit} samples, "
                    f"but only {len(items)} are available."
                )
            items = random.Random(seed).sample(items, limit)

        self.data = data
        self.image_col = image_col
        self.items = items
        self.tf = TRAIN_TF if train else TEST_TF

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        row_idx, label = self.items[i]
        image = self.data[row_idx][self.image_col]

        if not isinstance(image, Image.Image):
            raise TypeError(
                f"Expected a PIL image, got {type(image)}"
            )

        image = image.convert("RGB")
        return self.tf(image), label


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
