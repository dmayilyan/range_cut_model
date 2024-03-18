from dataclasses import dataclass, fields, is_dataclass
from typing import Callable


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
    learning_rate: float


@dataclass
class DnCNNConfig:
    files: Files
    params: Params

def get_fields(dc: Callable, field_dict: dict =  None):
    # We assume that we don't have repeating fields in our dataclasses
    if not field_dict:
        print(dc, field_dict)
        field_dict = {}

    for i in fields(dc):
        if is_dataclass(i.type):
            get_fields(i.type, field_dict)
        else:
            field_dict[i.name] = i.type.__name__
            print(field_dict)

    print("Before return")
    return field_dict

if __name__ == "__main__":
    field_dict = {}
    qwe = get_fields(DnCNNConfig, field_dict)
    print(qwe)

