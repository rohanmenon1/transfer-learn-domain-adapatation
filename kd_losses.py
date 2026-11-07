"""ERM and knowledge distillation (KD) losses.

Both take student logits zs [B, C], teacher logits zt [B, C] and labels y [B], and return (loss, parts dict).
"""
import torch.nn.functional as F


def _kl(zs, zt, T):
    return F.kl_div(F.log_softmax(zs / T, 1), F.softmax(zt / T, 1), reduction="batchmean") * T * T


def erm(zs, zt, y):
    ce = F.cross_entropy(zs, y)
    return ce, {"ce": ce}


def kd(zs, zt, y, T=1.0, w_ce=0.5, w_kd=0.5):
    """Hinton et al. 2015. ImageNet setting from mdistiller: T = 1, 0.5 CE + 0.5 KL."""
    ce, k = F.cross_entropy(zs, y), _kl(zs, zt, T)
    return w_ce * ce + w_kd * k, {"ce": ce, "kd": k}


LOSSES = {"erm": erm, "kd": kd}
