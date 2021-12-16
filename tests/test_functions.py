from src.functions import *
import numpy as np
from pytest import approx


def test_balance():
    assert balance(np.array([3, 1, 2, 3])).tolist() == approx([2, 1, 0])
    assert balance(np.array([4, 1, 2, 3])).tolist() == approx([2 + 1/3, 1 + 1/3, 1/3])
    assert balance(np.array([0.5, 1, 2, 3])).tolist() == approx([0.5, 0, 0])
    assert balance(np.array([1, 0, 0])).tolist() == approx([0.5, 0.5])
    assert balance(np.array([1, 0])).tolist() == approx([1])


def test_get_neighbors():
    neighbors = get_neighbors(np.array([
        [1, 2],
        [3, 4]
    ]))

    assert neighbors[0, 0].tolist() == [[1, 1, 2], [1, 1, 2], [3, 3, 4]]
    assert neighbors[1, 0].tolist() == [[1, 1, 2], [3, 3, 4], [3, 3, 4]]
    assert neighbors[0, 1].tolist() == [[1, 2, 2], [1, 2, 2], [3, 4, 4]]
    assert neighbors[1, 1].tolist() == [[1, 2, 2], [3, 4, 4], [3, 4, 4]]


def test_sum_neighbor_grid():
    foo = np.array([
        [1, 2],
        [3, 4]
    ])
    assert sum_neighbor_grid(get_neighbors(foo)).tolist() == (4 * foo).tolist()


def test_flatten():
    foo = np.array([
        [1, 2],
        [3, 4]
    ])
    assert flatten_sort(get_neighbors(foo))[0].tolist() == [
        [1, 1, 1, 1, 2, 2, 3, 3, 4],
        [1, 1, 2, 2, 2, 2, 3, 4, 4],
        [1, 1, 2, 3, 3, 3, 3, 4, 4],
        [1, 2, 2, 3, 3, 4, 4, 4, 4]
    ]


def test_flatten_unflatten():
    foo = np.array([
        [1, 2],
        [3, 4]
    ])
    assert unflatten_unsort(*flatten_sort(get_neighbors(foo))).tolist() == get_neighbors(foo).tolist()
