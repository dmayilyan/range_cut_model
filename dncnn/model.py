from torch import nn


class DnCNN(nn.Module):
    def __init__(self, number_of_layers=9, kernel_size=3):
        super(DnCNN, self).__init__()

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

    def forward(self, x):
        return self.dncnn(x)
