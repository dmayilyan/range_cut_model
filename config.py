from dataclasses import dataclass


@dataclass
class Files:
    name: str
    train_noisy: str
    train_sharp: str
    #  test_noisy: str
    #  test_sharp: str


@dataclass
class Params:
    epoch_count: int
    hidden_count: int


@dataclass
class DnCNNConfig:
    files: Files
    params: Params
