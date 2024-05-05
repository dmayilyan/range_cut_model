import logging

import torch
import torch.nn.functional as f
from torch import nn

logger = logging.getLogger(__name__)


class DnCNN(torch.nn.Module):
    def __init__(self, number_of_layers=9, kernel_size=3):
        super(DnCNN, self).__init__()

        # by default equals 1
        padding = int((kernel_size - 1) / 2)
        alpha = 0.2

        channels = 1
        features = 60

        layers = []
        layers.append(
            nn.Conv2d(
                in_channels=channels,
                out_channels=features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )
        layers.append(nn.LeakyReLU(negative_slope=alpha, inplace=True))

        for _ in range(number_of_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=kernel_size,
                    padding=padding,
                    bias=False,
                )
            )
            layers.append(nn.ReLU(inplace=True))
        layers.append(
            nn.Conv2d(
                in_channels=features,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            )
        )

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.dncnn(x)


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        size = 30  # Size of the matrix
        sigma = 60   # Standard deviation of the Gaussian
        self.weight_kernel = self.gaussian_2d(size, sigma).to("cuda")

    def gaussian_2d(self, size, sigma=1):
        # Create a grid
        x = torch.arange(size, dtype=torch.float32)
        y = torch.arange(size, dtype=torch.float32)
        x, y = torch.meshgrid(x, y)
        x = x - (size - 1) / 2
        y = y - (size - 1) / 2

        # Create Gaussian distribution
        gaussian = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

        # Normalize the distribution
        # gaussian = gaussian / gaussian.sum()

        return gaussian


    def forward(self, output: torch.Tensor, target: torch.Tensor):
        return f.l1_loss(self.weight_kernel * output, self.weight_kernel * target)
