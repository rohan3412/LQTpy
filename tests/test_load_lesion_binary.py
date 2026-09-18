"""Pins load_lesion's `was_binary` flag to its original np.unique semantics.

The detection was rewritten from an np.unique sort to an elementwise test for
speed. These cases guard the equivalence: "binary" means every value is exactly
0 or 1, so integral values other than 0/1 and negatives are NOT binary.
"""
import numpy as np
import pytest

from lqtpy.io import load_lesion


def _reference_was_binary(data):
    data = np.nan_to_num(data)
    u = np.unique(data)
    return bool(np.array_equal(u, u.astype(bool).astype(u.dtype)))


def _lesion(data):
    return (data.astype(np.float64), np.diag([1.0, 1.0, 1.0, 1.0]))


def _base():
    d = np.zeros((12, 12, 12))
    d[4:8, 4:8, 4:8] = 1.0
    return d


def _prob():
    d = _base() * 0.9
    d[5, 5, 5] = 0.3
    return d


def _with(value):
    d = _base()
    d[0, 0, 0] = value
    return d


@pytest.mark.parametrize("name,data", [
    ("binary", _base()),
    ("probabilistic", _prob()),
    ("integral two", _with(2.0)),
    ("negative one", _with(-1.0)),
    ("nan becomes zero", _with(np.nan)),
])
def test_was_binary_matches_unique_semantics(name, data):
    _, was_binary = load_lesion(_lesion(data))
    assert was_binary == _reference_was_binary(data), name
