import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import random
import warnings
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import timm
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

warnings.filterwarnings("ignore", message=".*still experimental.*")
pa.jemalloc_set_decay_ms(0)

REPO = "wltjr1007/DomainNet"
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
        files = sorted(str(f) for f in Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=f"data/{split}-*")).glob(f"data/{split}-*.parquet"))
        table = pq.read_table(files, columns=["image", "label"], filters=[("domain", "=", DOMAINS.index(domain))])
        labels = table["label"].to_pylist()
        idx = list(range(len(labels)))
        if per_class:
            by_class = defaultdict(list)
            for i in idx:
                by_class[labels[i]].append(i)
            rng = random.Random(seed)
            idx = [i for c in sorted(by_class) for i in (by_class[c] if len(by_class[c]) <= per_class else rng.sample(by_class[c], per_class))]
        if limit:
            idx = random.Random(seed).sample(idx, limit)
        images = pc.struct_field(table["image"], "bytes")
        self.images, self.labels = images.cast(pa.large_binary()).take(idx) if len(idx) < len(labels) else images, [labels[i] for i in idx]
        self.tf = TRAIN_TF if train else TEST_TF

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.tf(Image.open(BytesIO(self.images[i].as_py())).convert("RGB")), self.labels[i]


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


@torch.no_grad()
def features(model, data):
    model.eval()
    with torch.autocast("cuda", torch.bfloat16):
        f, y = zip(*[(model.forward_head(model.forward_features(x.cuda()), pre_logits=True).float(), y.cuda()) for x, y in DataLoader(data, batch_size=256, num_workers=10)])
    model.train()
    return torch.cat(f), torch.cat(y)


def linear_probe(model, data):
    x, y = features(model, data)
    head = model.get_classifier()
    w = torch.zeros_like(head.weight, dtype=torch.float32, requires_grad=True)
    b = torch.zeros_like(head.bias, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=300, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ w.t() + b, y) + 1e-4 * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        head.weight.copy_(w)
        head.bias.copy_(b)
