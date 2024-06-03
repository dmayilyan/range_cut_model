from torch import nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_len = 30

        padd = (0, 0, 0)
        bias = False
        if self.data_len == 30:
            padd = (1, 1, 1)

        alpha = 0.2

        layers = []

        layers.append(
            nn.Conv3d(
                1, self.data_len, kernel_size=4, stride=2, bias=bias, padding=(1, 1, 1)
            )
        )
        layers.append(nn.BatchNorm3d(self.data_len))
        layers.append(nn.LeakyReLU(alpha))
        for i in range(3):
            layers.append(
                nn.Conv3d(
                    self.data_len * (2**i),
                    self.data_len * 2 * (2**i),
                    kernel_size=4,
                    stride=2,
                    bias=bias,
                    padding=(1, 1, 1),
                )
            )
            layers.append(nn.BatchNorm3d(self.data_len * 2 * (2**i)))
            layers.append(nn.LeakyReLU(alpha))

        layers.append(
            nn.Conv3d(
                self.data_len * 8, 1, kernel_size=4, stride=2, bias=bias, padding=padd
            )
        )
        layers.append(nn.Sigmoid())

        self.wgan = nn.Sequential(*layers)

        def forward(self, x) -> torch.Tensor:
            return self.wgan(x)

