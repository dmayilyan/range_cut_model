import hydra
from omegaconf import OmegaConf
from config import DnCNNConfig
from dncnn.utils import get_device
from dncnn.dataset import create_dataloader
from dncnn.model import DnCNN, Loss
from torch import optim
import numpy as np

import logging

logger = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DnCNNConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    device = get_device()

    model = DnCNN(number_of_layers=cfg.params.hidden_count, kernel_size=3).to(
        device=device
    )
    model.eval()
    data_loader = create_dataloader(
        root_path=cfg.paths.data,
        file_path_noisy=cfg.files.train_data_noisy,
        file_path_sharp=cfg.files.train_data_sharp,
    )

    criterion = Loss()
    criterion.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)

    training_losses = np.zeros(cfg.params.epoch_count)
    validation_losses = np.zeros(cfg.params.epoch_count)
    for epoch in range(cfg.params.epoch_count):
        train_loss = 0
        for i, data in enumerate(data_loader):
            model.train()
            model.zero_grad()

            data_cut_big, data_cut_small = data

            output = model((data_cut_big.float().to(device)))
            # plt.matshow(output[0].to("cpu").detach().numpy())
            # plt.colorbar()
            # display(output.size())
            # display(output.squeeze((0, 1, 2)).size)
            batch_loss = criterion(output.to(device), data_cut_small.to(device)).to(
                device
            )
            batch_loss.backward()
            optimizer.step()
            model.eval()
            train_loss += batch_loss.item()
        train_loss = train_loss / len(data)
        training_losses[epoch] = train_loss
        logger.info("train loss for epoch %d: %f", train_loss)


if __name__ == "__main__":
    main()
