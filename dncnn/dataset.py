import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler, SequentialSampler

from dncnn.load_data import load_data
from dncnn.utils import slice_to_shortest

logger = logging.getLogger(__name__)


class CaloData(Dataset[Any]):
    def __init__(self, data_noisy: np.ndarray, data_sharp: np.ndarray, transform=None):
        self.data_cut_big = torch.from_numpy(np.sum(data_noisy, axis=3))
        self.data_cut_small = torch.from_numpy(np.sum(data_sharp, axis=3))
        #  print(self.data_cut_big.size())
        self.transform = transform

    def __len__(self):
        # Needs to be address, currently non-functional

        #  logger.info(self.data_cut_big.shape)
        #  logger.info(self.data_cut_big.size())
        #  logger.info(len(self.data_cut_big))
        #  logger.info("- - -")
        return len(self.data_cut_big)

    def __getitem__(self, idx):
        #  print(self.data_cut_big[idx, :, :].size())
        return self.data_cut_big[idx, :, :], self.data_cut_small[idx, :, :]


def create_dataloader(
    root_path: str, file_path_noisy: str, file_path_sharp: str, train: bool=True
) -> DataLoader[Any]:
    data_path_noisy = Path(f"{root_path}/{file_path_noisy}")
    data_path_sharp = Path(f"{root_path}/{file_path_sharp}")
    data_noisy = load_data(data_path_noisy)
    data_sharp = load_data(data_path_sharp)
    # Slice events to the shortest
    data_noisy, data_sharp = slice_to_shortest(data_noisy, data_sharp)
    dataset = CaloData(data_noisy, data_sharp)

    tt_split = int(0.8 * len(dataset))
    indices = list(range(len(dataset)))

    if train:
        indices = indices[:tt_split]
    else:
        indices = indices[tt_split:]

    sampler = SequentialSampler(indices)

    return DataLoader(
        CaloData(data_noisy, data_sharp), sampler=sampler, num_workers=2
    )

if __name__ == "__main__":
    print("Will read the file now")
    data = create_dataloader(root_path="../ILDCaloSim",
                             file_path_noisy="e-_Jun3/test/showers-10kE10GeV-RC10-1.hdf5",
                             file_path_sharp="e-_Jun3/test/showers-10kE10GeV-RC01-1.hdf5",
                             train=False
    )
    print("Have read the file now")
    vals = []
    for i, val in enumerate(data):
        if i >= 100:
            break

        a, b = val
        a, b = a.squeeze(), b.squeeze()
        #  print(a.size())
        #  print(b.size())

        vals += a.flatten().tolist()
        #  print(a.flatten().tolist())
        print(i, a.flatten().min())
    import matplotlib.pyplot as plt
    plt.hist(vals, bins=np.arange(-10, 14, 0.5))
    plt.semilogy()
    plt.savefig("test.png")
    plt.figure()
    plt.hist(np.log(vals))
    plt.semilogy()
    plt.savefig("test_log.png")

