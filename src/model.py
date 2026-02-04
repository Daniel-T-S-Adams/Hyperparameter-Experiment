import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        layers,
        dropout: float,
        leaky_relu_alpha: float,
        input_dim: int = 28 * 28,
        num_classes: int = 10,
    ):
        super().__init__()
        modules = []
        prev_dim = input_dim
        for width in layers:
            modules.append(nn.Linear(prev_dim, width))
            modules.append(nn.LeakyReLU(negative_slope=leaky_relu_alpha))
            modules.append(nn.Dropout(p=dropout))
            prev_dim = width
        modules.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        return self.net(x)
