import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler, SequentialSampler

#  from sklearn.preprocessing import power_transform
#  sklearn.preprocessing.PowerTransformer
from dncnn.load_data import load_data
from dncnn.utils import slice_to_shortest

logger = logging.getLogger(__name__)

from scipy.stats import yeojohnson


class CaloData(Dataset[Any]):
    def __init__(self, data_noisy: np.ndarray, data_sharp: np.ndarray, transform=None):
        self.data_sharp = torch.from_numpy(data_sharp)

        #  self.data_sharp = torch.sum(self.data_sharp, dim=3)

        self.transform = transform

        #  self.data_noisy = torch.sqrt(self.data_noisy)
        #  self.data_sharp = torch.sqrt(self.data_sharp)

        print(self.data_sharp.shape)
        mins_sharp = torch.amin(self.data_sharp, dim=(1, 2, 3), keepdim=True)
        maxes_sharp = torch.amax(self.data_sharp, dim=(1, 2, 3), keepdim=True)

        self.data_sharp = (self.data_sharp - mins_sharp)/(maxes_sharp-mins_sharp)

        #  self.mean_noisy = torch.mean(self.data_noisy, dim=(1, 2), keepdim=True)
        #  self.std_noisy = torch.std(self.data_noisy, dim=(1, 2), keepdim=True)
        #  self.data_noisy = (self.data_noisy - self.mean_noisy) / self.std_noisy

        #  print(mean_noisy.shape)
        #  #  print(f"{mean_noisy.shape=} {std_noisy.shape=}")

        #  mean_sharp = torch.mean(self.data_sharp, dim=(1, 2, 3), keepdim=True)
        #  std_sharp = torch.std(self.data_sharp, dim=(1, 2, 3), keepdim=True)
        #  self.data_sharp = (self.data_sharp - mean_sharp) / std_noisy

        #  logger.info(f"data_noisy before sum: {self.data_noisy}")


        #  for i, x in enumerate(torch.unbind(self.data_noisy, dim=0)):
        #  if i > 1:
        #  break
        #  print(x.shape)
        #  self.data_noisy = torch.stack([torch.tensor(yeojohnson(x)) for x in torch.unbind(self.data_noisy, dim=0)], dim=0)
        #  self.data_noisy = torch.stack([torch.tensor(yeojohnson(x.flatten().tolist())[0]).reshape(30, 30) for x in torch.unbind(self.data_noisy, axis=0)])
        #  self.data_sharp = torch.stack([torch.tensor(yeojohnson(x.flatten().tolist())[0]).reshape(30, 30) for x in torch.unbind(self.data_sharp, axis=0)])
        #  self.data_sharp = torch.stack([yeojohnson(x.tolist()) for x in torch.unbind(self.data_sharp, dim=0)], dim=0)
        #  print(f"{mean_sharp.shape=} {std_sharp.shape=}")

    def __len__(self):
        #  logger.warning("DEBUG data")
        # Needs to be address, currently non-functional
        return len(self.data_sharp)# // 18

    def __getitem__(self, idx):
        return self.data_sharp[idx, :, :, :]

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


    tt_split = int(0.75 * len(dataset))
    #  indices = list(range(len(dataset)))
    random.seed(42)

    indices = random.sample(range(len(dataset)), len(dataset))

    if is_train:
        indices = indices[:tt_split]
    else:
        indices = indices[tt_split:]

    sampler = SequentialSampler(indices)

    return DataLoader(
        CaloData(data_noisy, data_sharp, transform),
        sampler=sampler,
        #  batch_size=2,
        num_workers=2,
    )


if __name__ == "__main__":
    print("Will read the file now")
    #  with open("train_loader.pkl", "rb") as f:
    #  logging.info("Reading train_loader from a pickle.")
    #  train_loader = pickle.load(f)
    train_loader = create_dataloader(
        root_path="../ILDCaloSim",
        file_path_noisy="e-_Jun3/test/showers-10kE10GeV-RC10-1.hdf5",
        file_path_sharp="e-_Jun3/test/showers-10kE10GeV-RC01-1.hdf5",
        is_train=False,
    )

    train_loader = create_dataloader(
        root_path="../ILDCaloSim",
        file_path_noisy="e-_Jun3/test/showers-10kE10GeV-RC10-1.hdf5",
        file_path_sharp="e-_Jun3/test/showers-10kE10GeV-RC01-1.hdf5",
        is_train=True,
    )
