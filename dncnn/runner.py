import logging
from typing import Any

import torch
from torch.utils.data.dataloader import DataLoader

from dncnn.model import Loss

logger = logging.getLogger(__name__)

class Runner:
    def __init__(
        self,
        data_loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
    ) -> None:
        self.data_loader = data_loader
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss = Loss()

    def run(self):
        self.model.train()
        self.model.zero_grad()
        self.loss.to(self.device)

        loss_sum = 0
        for data_cut_noisy, data_cut_sharp in self.data_loader:
            loss_val = self.run_single(data_cut_noisy, data_cut_sharp)

            loss_sum += loss_val

            #  self.loss.backward()
            self.optimizer.step()
            self.model.eval()

        #  logger.info("train loss for epoch %d: %f", epoch + 1, train_loss)
        loss_average = loss_sum / len(self.data_loader)

    def run_single(self, data_cut_noisy: torch.Tensor, data_cut_sharp: torch.Tensor):
        data_cut_noisy = data_cut_noisy.float().to(self.device)
        data_cut_sharp = data_cut_sharp.float().to(self.device)
        # shape is 1x30x30
        prediction = self.model(data_cut_noisy)

        loss = self.loss(prediction, data_cut_sharp)

        del prediction
        del data_cut_noisy
        del data_cut_sharp

        return loss


def run_epoch(train_runner: Runner, epoch_id: int) -> None:
    logger.info("Running epoch %d", epoch_id)
    train_runner.run()
