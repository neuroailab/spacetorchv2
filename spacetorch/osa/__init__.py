"""
Code in this directory has been taken verbatim, for most part, from publicly-accessable https://github.com/anonymous2022icml/ViTViS.

Preprint:
Ghiasi, Amin, Hamid Kazemi, Eitan Borgnia, Steven Reich, Manli Shu, Micah Goldblum, Andrew Gordon Wilson, and Tom Goldstein. "What Do Vision Transformers Learn? A Visual Exploration." ArXiv preprint arXiv:2212.06727 (2022).
"""

import collections
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from spacetorch.osa.loss import LossArray


def new_init(size: int, batch_size: int = 1, last: torch.nn = None, padding: int = -1, zero: bool = False) -> torch.nn:
    output = torch.rand(size=(batch_size, 3, size, size)) if not zero else torch.zeros(size=(batch_size, 3, size, size))
    # output += 0.5
    output = output.cuda()
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).cuda()
        scaled = up(last)
        cx = (output.patch_size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output


class Optimize:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None, post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, **_):
        self.loss = loss_array

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None):
        img = img.detach().clone().to('cuda:0').requires_grad_()

        optimizer = optimizer if optimizer is not None else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        for i in tqdm(range(self.steps + 1)):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()
            torch.cuda.empty_cache()

        optimizer.state = collections.defaultdict(dict)
        return img
