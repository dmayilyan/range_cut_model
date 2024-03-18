import logging
import pickle

from functools import wraps
import hydra
from hydra.core.config_store import ConfigStore
from torch import optim

from config import DnCNNConfig
from dncnn.model import DnCNN  # , Loss
from dncnn.runner import Runner, run_epoch
from dncnn.utils import get_device

from dncnn.model import Loss
import numpy as np

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="dncnn_config", node=DnCNNConfig)

from functools import wraps
from typing import Callable

from omegaconf import DictConfig, OmegaConf

def mark_for_write(*args, **kwargs):
    def inner_decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapped(cfg: DictConfig):
            print(OmegaConf.to_yaml(cfg))  # do some stuff
            #  if "db_name" not in kwargs:
                #  cfg["db_cursor"] = "qwre"
            return func(cfg) # pass cfg to decorated function
        wrapped.db_cursor = "123123"
        print('decorating', func, 'with argument', ", ".join([i for i in args]), "|", ", ".join([i for i in kwargs]))
        return wrapped

    return inner_decorator


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
@mark_for_write(db_name="gweqwe")
def main(cfg: DnCNNConfig) -> None:
    logger.info(dir())
    device = get_device()
    logger.info("Before return")
    return

    model = DnCNN(number_of_layers=cfg.params.hidden_count, kernel_size=3).to(
    #  model = DnCNN(number_of_layers=3, kernel_size=3).to(
        device=device
    )
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)

    #  model.eval()
    #  data_loader = create_dataloader(
    #  root_path=cfg.paths.data,
    #  file_path_noisy=cfg.files.train_data_noisy,
    #  file_path_sharp=cfg.files.train_data_sharp,
    #  )
    #  with open("data_loader.pkl", "wb") as f:
    #  pickle.dump(data_loader, f)

    with open("data_loader.pkl", "rb") as f:
        logging.info("Reading from a pickle.")
        data_loader = pickle.load(f)

    #  runner = Runner(data_loader, model, optimizer, device)

    criterion = Loss()
    criterion.to(device)

    training_losses = np.zeros(cfg.params.epoch_count)
    validation_losses = np.zeros(cfg.params.epoch_count)
    for epoch in range(cfg.params.epoch_count):
        #  run_epoch(runner, epoch)
        #  logger.info("train loss for epoch %d: %f", epoch + 1, train_loss)

        train_loss = 0
        for i, data in enumerate(data_loader):
            model.train()
            model.zero_grad()

            data_cut_big, data_cut_small = data

            output = model((data_cut_big.float().to(device)))
        #  # plt.matshow(output[0].to("cpu").detach().numpy())
        #  # plt.colorbar()
        #  # display(output.size())
        #  # display(output.squeeze((0, 1, 2)).size)
            batch_loss = criterion(output.to(device), data_cut_small.to(device)).to(
            device
            )
            batch_loss.backward()
            optimizer.step()
            model.eval()
            train_loss += batch_loss.item()
            train_loss = train_loss / len(data)
            training_losses[epoch] = train_loss
        logger.info("train loss for epoch %d: %f", epoch + 1, train_loss)


if __name__ == "__main__":
    main()
