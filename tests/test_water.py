import numpy as np
from pytest import raises
from src.water import Water


def test_init():
    arr = np.array([
        [1, 2],
        [3, 4]
    ])
    foo = Water(arr)

    assert foo.dem.tolist() == arr.tolist()
    assert np.all(foo.slope > 0)

    assert foo.direction[1, 1, :, :].tolist() == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert foo.direction[1, 0, :, :].tolist() == [
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


def test_errs():
    with raises(ValueError):
        Water(np.array([]))

    with raises(ValueError):
        Water(np.array([[]]))


def test_step():
    arr = np.array([
        [1, 2],
        [3, 4]
    ])
    foo = Water(arr)

    assert foo.depth.tolist() == [[0, 0], [0, 0]]
    foo.step()
    assert foo.depth.tolist() == [[0, 0], [0, 0]]

    foo.source = np.array([
        [0, 0],
        [0, 1]]
    )

    foo.step()
    assert (foo.depth > 0).tolist() == [
        [False, False],
        [False, True]
    ]

    foo.step()
    print(foo.depth > 0)
    assert (foo.depth > 0).tolist() == [
        [True, False],
        [False, True]
    ]
