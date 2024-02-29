import torch
import logging

logger = logging.getLogger(__name__)


def get_device() -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.warning("Device is set to '%s'", device)

    return device
