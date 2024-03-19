from dataclasses import dataclass, fields, is_dataclass
from typing import Callable


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


def get_fields(dc: Callable, field_dict: dict = None):
    # We assume that we don't have repeating fields in our dataclasses
    for i in fields(dc):
        if is_dataclass(i.type):
            get_fields(i.type, field_dict)
        else:
            field_dict[i.name] = i.type.__name__

    return field_dict


if __name__ == "__main__":
    field_dict = {}
    qwe = get_fields(DnCNNConfig, field_dict)
    print("- - -")
    #  print(f"{field_dict=}")
    print(qwe)
