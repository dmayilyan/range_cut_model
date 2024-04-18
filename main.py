import logging
from uuid import uuid4
import pickle
from functools import wraps
from sqlite3 import Cursor, connect

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch import optim
from torch.utils.data import random_split

from config import DnCNNConfig, LogConfig
from dncnn.dataset import create_dataloader
from dncnn.model import DnCNN, Loss
from dncnn.utils import flatten_dict, get_device, get_fields

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="config_scheme", node=DnCNNConfig)

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

        return func(*args, **kwargs)

    return wrapped


# log writing needs refactoring
@hydra.main(config_path="conf", config_name="config", version_base="1.3")
@mark_for_write
def main(
    cfg: DnCNNConfig, db_name: str | None = None, db_cur: Cursor | None = None
) -> None:
    print(cfg)

    field_dict = {}
    get_fields(LogConfig, field_dict)
    fields_str = ", ".join(f"{colval[0]} {colval[1]}" for colval in field_dict.items())
    db_cur.execute(
    f"CREATE TABLE IF NOT EXISTS {LogConfig.__name__} ({fields_str}, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )

    main_dict = OmegaConf.to_object(cfg)
    print(type(main_dict), main_dict)

    flattened_main_dict = flatten_dict(main_dict)
    columns, values = flattened_main_dict.keys(), flattened_main_dict.values()
    columns = list(columns)
    values = tuple(values)

    device = get_device()

    model = DnCNN(number_of_layers=cfg.params.hidden_count, kernel_size=cfg.params.kernel_size).to(
        device=device
    )
    model.eval()
    #  optimizer = optim.Adam(model.parameters(), lr=cfg.params.learning_rate, weight_decay=cfg.params.weight_decay)
    optimizer = optim.Adam(
        model.parameters(), lr=cfg.params.learning_rate, weight_decay=0.001
    )

    train_loader = create_dataloader(
        root_path=cfg.files.root_path,
        file_path_noisy=cfg.files.train_noisy,
        file_path_sharp=cfg.files.train_sharp,
        is_train=True,
    )

    test_loader = create_dataloader(
        root_path=cfg.files.root_path,
        file_path_noisy=cfg.files.train_noisy,
        file_path_sharp=cfg.files.train_sharp,
        is_train=False,
        transform=train_loader.dataset.transform,
    )

    #  with open("train_loader.pkl", "wb") as f:
    #  pickle.dump(train_loader, f)

    #  with open("test_loader.pkl", "wb") as f:
    #  pickle.dump(test_loader, f)

    #  return

    #  with open("train_loader.pkl", "rb") as f:
    #  logging.info("Reading train_loader from a pickle.")
    #  train_loader = pickle.load(f)

    #  with open("test_loader.pkl", "rb") as f:
    #  logging.info("Reading test_loader from a pickle.")
    #  test_loader = pickle.load(f)

    criterion = Loss()
    criterion.to(device)

    run_id = uuid4().fields[-1]
    logger.info("Run id %s", str(run_id))
    columns += ["run_id", "epoch", "train_loss", "test_loss"]
    training_losses = np.zeros(cfg.params.epoch_count)
    test_losses = np.zeros(cfg.params.epoch_count)
    for epoch in range(cfg.params.epoch_count):
        train_loss = 0
        train_max_s = []
        train_max_b = []
        for i, data in enumerate(train_loader):
            #  print("epoch", epoch, "train", i)
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

        #  print(f"{max(train_max_s)=} {mean(train_max_s)=}")
        #  print(f"{max(train_max_b)=} {mean(train_max_b)=}")
        #  print(f"{len(data)=}")
        #  train_loss = train_loss / len(data)
        train_loss = train_loss / i
        #  print("train i", i, train_loss)
        training_losses[epoch] = train_loss

        test_loss = 0
        for i, data in enumerate(test_loader):
            data_cut_big, data_cut_small = data

            #  logging.info("test data_cut big", data_cut_big.shape)
            #  logging.info("data_cut_small", data_cut_small.shape)

            output = model((data_cut_big.float().to(device)))
            batch_loss = criterion(output.to(device), data_cut_small.to(device)).to(
                device
            )
            test_loss += batch_loss.item()
        #  test_loss = test_loss / len(data)
        test_loss = test_loss / i
        #  print("test i", i, test_loss)
        test_losses[epoch] = test_loss

        db_cur.execute(
        f"""INSERT INTO {LogConfig.__name__} ({", ".join(columns)}) VALUES ({','.join('?'*len(values + (run_id, epoch, train_loss, test_loss)))})""",
        values + (run_id, epoch, train_loss, test_loss),
        )
        logger.info(
            "epoch %3d: train_loss: %f, test_loss: %f", epoch, train_loss, test_loss,
        )

    torch.save(model.state_dict(), f"./models/{run_id}.torch")


if __name__ == "__main__":
    main()
