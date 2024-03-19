import logging
import pickle
from dataclasses import fields
from functools import wraps
from sqlite3 import Cursor, connect

import hydra
import numpy as np
from hydra.core.config_store import ConfigStore
from torch import optim

from config import DnCNNConfig, Params
from dncnn.model import DnCNN, Loss
from dncnn.utils import get_device
from dncnn.dataset import create_dataloader

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="dncnn_config", node=DnCNNConfig)

from typing import Callable

from omegaconf import OmegaConf


def mark_for_write(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(*args, **kwargs):
        if "db_name" not in kwargs:
            db_name = "log.db"
            kwargs["db_name"] = db_name
        else:
            db_name = kwargs["db_name"]

        conn = connect(db_name, isolation_level=None)
        db_cur = conn.cursor()
        kwargs["db_cur"] = db_cur

        # Create the table based on Params dataclass
        fields_str = ", ".join(
            f"{field.name} {field.type.__name__}" for field in fields(Params)
        )
        db_cur.execute(f"CREATE TABLE IF NOT EXISTS {Params.__name__} ({fields_str})")

        return func(*args, **kwargs)  # pass cfg to decorated function

    return wrapped


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
@mark_for_write
def main(
    cfg: DnCNNConfig, db_name: str | None = None, db_cur: Cursor | None = None
) -> None:
    cfg_dict = OmegaConf.to_object(cfg)
    logger.info(OmegaConf.to_yaml(cfg))

    columns, values = cfg_dict["params"].keys(), cfg_dict["params"].values()
    columns = ", ".join(map(str, columns))
    values = ", ".join(map(str, values))

    db_cur.execute(f"INSERT INTO Params ({columns}) VALUES ({values})")
    logging.info("Logging '%s' into '%s'", values, columns)

    device = get_device()

    model = DnCNN(number_of_layers=cfg.params.hidden_count, kernel_size=3).to(
        device=device
    )
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate)

    #  model.eval()
    data_loader = create_dataloader(
    root_path=cfg.files.root_path,
    file_path_noisy=cfg.files.train_data_noisy,
    file_path_sharp=cfg.files.train_data_sharp,
    )
    with open("data_loader.pkl", "wb") as f:
        pickle.dump(data_loader, f)

    return
    #  with open("data_loader.pkl", "rb") as f:
        #  logging.info("Reading from a pickle.")
        #  data_loader = pickle.load(f)

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
            #  db_cur.execute("INSERT INTO Params (train_loss) VALUES ()", (train_loss))
            training_losses[epoch] = train_loss
        logger.info("train loss for epoch %d: %f", epoch + 1, train_loss)


if __name__ == "__main__":
    main()
