import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.warning("Device is set to '%s'", device)

    return device


def slice_to_shortest(np1, np2, axis=0) -> tuple[np.ndarray, np.ndarray]:
    np1_shape = np1.shape
    np2_shape = np2.shape

    min_in_axis = min(np1_shape[axis], np2_shape[axis])

    return np1.take(range(0, min_in_axis), axis=axis), np2.take(
        range(0, min_in_axis), axis=axis
    )
