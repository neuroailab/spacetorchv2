import torch
import torch.nn as nn
import numpy as np

from spacetorch.osa.regularizers import TotalVariation as BaseTotalVariation


_nums = '0123456789'


def _abbreviation(name: str) -> str:
    if len(name) <= 3:
        return name
    abr = ''.join(x for x in name if x.isupper() or x in _nums)
    return abr[:3]


def _round(num: float) -> str:
    if num > 100:
        return str(int(round(num, 0)))
    if num > 10:
        return str(round(num, 1))
    return str(round(num, 2))


class InvLoss:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = _abbreviation(self.__class__.__name__)
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        self.last_value = tensor.item()
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f'{_round(self.c * self.last_value)}({_round(self.last_value)})'

    def reset(self) -> torch.tensor:
        return 0
    

class LossArray:
    def __init__(self):
        self.losses = []
        self.last_value = 0

    def __add__(self, other: InvLoss):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor):
        tensor = sum(l(x) for l in self.losses)
        self.last_value = tensor.item()
        return tensor

    def header(self) -> str:
        rest = '\t'.join(l.name for l in self.losses)
        return f'Loss\t{rest}'

    def __str__(self):
        rest = '\t'.join(str(l) for l in self.losses)
        return f'{_round(self.last_value)}\t{rest}'

    def reset(self):
        return sum(l.reset() for l in self.losses)


class TotalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size


class ViTFeatHook(InvLoss):
    def __init__(self, hook: nn.Module, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, :mn].diag().mean()


class ViTEnsFeatHook(ViTFeatHook):
    def __init__(self, hook: nn.Module, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()
