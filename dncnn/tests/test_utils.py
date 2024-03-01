import pytest
from ..utils import slice_to_shortest
import numpy as np

def test_slice_to_shortest():
    print(np.empty((2, 3, 5)).round(2))
    assert 1 == 0


if __name__ == "__main__":
    test_slice_to_shortest()
