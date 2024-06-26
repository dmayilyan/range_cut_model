from dataclasses import dataclass


@dataclass
class Files:
    root_path: str
    train_noisy: str
    train_sharp: str


@dataclass
class Params:
    epoch_count: int
    weight_decay: float
    kernel_size: int
    hidden_count: int
    learning_rate: float


@dataclass
class DnCNNConfig:
    files: Files
    params: Params

@dataclass
class WGANConfig:
    files: Files

@dataclass
class LogConfig:
    run_id: int
    train_loss: float
    test_loss: float
    epoch: int
    model_params: DnCNNConfig
