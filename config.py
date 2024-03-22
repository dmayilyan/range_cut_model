from dataclasses import dataclass


@dataclass
class Files:
    root_path: str
    train_noisy: str
    train_sharp: str
    #  test_noisy: str
    #  test_sharp: str


@dataclass
class Params:
    epoch_count: int
    hidden_count: int
    learning_rate: float


@dataclass
class DnCNNConfig:
    files: Files
    params: Params


@dataclass
class LogConfig:
    train_loss: float
    test_loss: float
    epoch: int
    model_params: DnCNNConfig


if __name__ == "__main__":
    #  field_dict = {}
    #  qwe = get_fields(DnCNNConfig, field_dict)
    #  print("- - -")
    #  #  print(f"{field_dict=}")
    #  print(qwe)
    print(LogConfig.__name__)
