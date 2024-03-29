import numpy as np
import pytest

from ..utils import flatten_dict, slice_to_shortest


@pytest.fixture
def arrays():
    arr1 = np.empty((2, 5)).round(2)
    arr2 = np.empty((5, 6)).round(2)
    return arr1, arr2


@pytest.fixture
def nested_dict():
    nested_dict = {
        "key0": {"key1": "value1"},
        "key2": {"key3": "value3", "key4": "value4", "key5": "value5"},
    }

    return nested_dict


def test_slice_to_shortest_axis_default(arrays):
    arr1, arr2 = arrays[0], arrays[1]
    sliced_arr1, sliced_arr2 = slice_to_shortest(arr1, arr2)
    min_axis0 = min(arr1.shape[0], arr2.shape[0])

    assert sliced_arr1.shape[0] == min_axis0
    assert sliced_arr2.shape[0] == min_axis0


def test_slice_to_shortest_axis_1(arrays):
    arr1, arr2 = arrays[0], arrays[1]
    sliced_arr1, sliced_arr2 = slice_to_shortest(arr1, arr2, axis=1)
    min_axis1 = min(arr1.shape[1], arr2.shape[1])

    assert sliced_arr1.shape[1] == min_axis1
    assert sliced_arr2.shape[1] == min_axis1


def test_flatten_dict(nested_dict):
    flat_result = flatten_dict(nested_dict)
    expected_result = {
        "key1": "value1",
        "key3": "value3",
        "key4": "value4",
        "key5": "value5",
    }

    assert flat_result == expected_result
