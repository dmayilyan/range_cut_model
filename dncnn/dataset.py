import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler

from dncnn.load_data import load_data
from dncnn.utils import slice_to_shortest

logger = logging.getLogger(__name__)


class CaloData(Dataset[Any]):
    def __init__(self, data_noisy: np.ndarray, data_sharp: np.ndarray, transform=None):
        self.data_noisy = torch.from_numpy(data_noisy)
        self.data_sharp = torch.from_numpy(data_sharp)

        print("data_noisy.shape", self.data_noisy.shape)
        self.transform = transform

        if self.transform is None:
            mean_noisy = torch.mean(self.data_noisy, dim=(1, 2, 3), keepdim=True)
            std_noisy = torch.std(self.data_noisy, dim=(1, 2, 3), keepdim=True)
            print(f"{mean_noisy.shape=} {std_noisy.shape=}")

            mean_sharp = torch.mean(self.data_sharp, dim=(1, 2, 3), keepdim=True)
            std_sharp = torch.std(self.data_sharp, dim=(1, 2, 3), keepdim=True)
            print(f"{mean_sharp.shape=} {std_sharp.shape=}")

            self.transform = {
                "noisy": {"mean": mean_noisy, "std": std_noisy},
                "sharp": {"mean": mean_sharp, "std": std_sharp},
            }

    def __len__(self):
        # Needs to be address, currently non-functional
        return len(self.data_noisy)

    def __getitem__(self, idx):
        if self.transform:
            self.data_noisy = (
                self.data_noisy - self.transform["noisy"]["mean"]
            ) / self.transform["noisy"]["std"]
            self.data_sharp = (
                self.data_sharp - self.transform["sharp"]["mean"]
            ) / self.transform["sharp"]["std"]

            print("normalized",
                  self.data_noisy.shape,
                  torch.mean(self.data_noisy, dim=(1, 2, 3)),
                  torch.mean(self.data_noisy, dim=(1, 2, 3)).shape)

            print("sum shape",
                torch.sum(self.data_noisy, dim=3).shape)

            return (
                torch.sum(self.data_noisy, dim=3),
                torch.sum(self.data_sharp, dim=3),
            )
        else:
            logger.error("Transform was not specified.")

    #  def _prepare_transformations(self, data_noisy, data_sharp):


def create_dataloader(
    root_path: str,
    file_path_noisy: str,
    file_path_sharp: str,
    is_train: bool = True,
    transform: dict = None,
) -> DataLoader[Any]:
    data_path_noisy = Path(f"{root_path}/{file_path_noisy}")
    data_path_sharp = Path(f"{root_path}/{file_path_sharp}")
    data_noisy = load_data(data_path_noisy)
    data_sharp = load_data(data_path_sharp)
    # Slice events to the shortest
    data_noisy, data_sharp = slice_to_shortest(data_noisy, data_sharp)
    dataset = CaloData(data_noisy, data_sharp, transform)

    tt_split = int(0.8 * len(dataset))
    indices = list(range(len(dataset)))

    if is_train:
        indices = indices[:tt_split]
    else:
        indices = indices[tt_split:]

    sampler = SequentialSampler(indices)

    return DataLoader(
        CaloData(data_noisy, data_sharp, transform), sampler=sampler, num_workers=2
    )


if __name__ == "__main__":
    print("Will read the file now")
    data = create_dataloader(
        root_path="../ILDCaloSim",
        file_path_noisy="e-_Jun3/test/showers-10kE10GeV-RC10-1.hdf5",
        file_path_sharp="e-_Jun3/test/showers-10kE10GeV-RC01-1.hdf5",
        train=False,
    )
    print("Have read the file now")
    vals = []
    for i, val in enumerate(data):
        if i >= 100:
            break

        a, b = val
        #  a, b = a.squeeze(), b.squeeze()
        #  print(a.size())
        #  print(b.size())

        vals += a.flatten().tolist()
        #  print(a.flatten().tolist())
        print(i, a.flatten().min())
    import matplotlib.pyplot as plt

    plt.hist(vals, bins=np.arange(-10, 14, 0.5))
    plt.semilogy()
    plt.savefig("test.png")
    #  plt.figure()
    #  plt.hist(np.log(vals))
    #  plt.semilogy()
    #  plt.savefig("test_log.png")
