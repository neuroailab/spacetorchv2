import torch
import torch.nn as nn


class TotalVariation(nn.Module):
    def __init__(self, p: int = 2):
        super().__init__()
        self.p = p

    def forward(self, x: torch.tensor) -> torch.tensor:
        x_wise = x[:, :, :, 1:] - x[:, :, :, :-1]
        y_wise = x[:, :, 1:, :] - x[:, :, :-1, :]
        diag_1 = x[:, :, 1:, 1:] - x[:, :, :-1, :-1]
        diag_2 = x[:, :, 1:, :-1] - x[:, :, :-1, 1:]
        return x_wise.norm(p=self.p, dim=(2, 3)).mean() + y_wise.norm(p=self.p, dim=(2, 3)).mean() + \
               diag_1.norm(p=self.p, dim=(2, 3)).mean() + diag_2.norm(p=self.p, dim=(2, 3)).mean()
