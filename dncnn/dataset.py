from pathlib import Path

from typing import Any
import torch
from torch.utils.data import DataLoader, Dataset
from dncnn.load_data import load_data
from dncnn.utils import slice_to_shortest
import numpy as np

import logging

logger = logging.getLogger(__name__)


class CaloData(Dataset):
    def __init__(self, data_noisy: np.ndarray, data_sharp: np.ndarray, transform=None):
        self.data_cut_big = torch.from_numpy(np.sum(data_noisy, axis=3))
        self.data_cut_small = torch.from_numpy(np.sum(data_sharp, axis=3))
        self.transform = transform

    def __len__(self):
        return len(self.data_cut_big)

    def __getitem__(self, idx):
        # event_idx = self.data[idx]

        # if self.transform:
        #     sample = self.transform(self.data)

        # print(type(sample))

        return self.data_cut_big[idx, :, :], self.data_cut_small[idx, :, :]


def create_dataloader(
    root_path: str, file_path_noisy: str, file_path_sharp: str
) -> DataLoader[Any]:
    data_path_noisy = Path(f"{root_path}/{file_path_noisy}")
    data_path_sharp = Path(f"{root_path}/{file_path_sharp}")
    data_noisy = load_data(data_path_noisy)
    data_sharp = load_data(data_path_sharp)
    # Slice events to the shortest
    data_noisy, data_sharp = slice_to_shortest(data_noisy, data_sharp)
    return DataLoader(dataset=CaloData(data_noisy, data_sharp), num_workers=2)
