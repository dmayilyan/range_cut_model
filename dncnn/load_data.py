import logging

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def load_data(data_path) -> np.ndarray:
    logger.info("Loading data from: %s", data_path)
    data = h5py.File(data_path, "r")["30x30"]["layers"]
    data = np.array(data)
    logger.info("Loaded '%s' with size %s", data_path, str(data.shape))

    return data
