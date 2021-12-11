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


def test_errs():
    with raises(ValueError):
        Water(np.array([]))

    with raises(ValueError):
        Water(np.array([[]]))


def test_flow_direction():
    foo = Water([
        [1, 2],
        [3, 4]
    ])

    assert foo.direction[0, 0].tolist() == [
        [1/3, 1/3, 0],
        [1/3, 0, 0],
        [0, 0, 0]
    ]

    assert foo.direction[1, 0].tolist() == [
        [0.5, 0.5, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]

    assert foo.direction[0, 1].tolist() == [
        [1 / 2, 0, 0],
        [1 / 2, 0, 0],
        [0, 0, 0]
    ]

    assert foo.direction[1, 1].tolist() == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]


def test_step():
    foo = Water([
        [1, 2],
        [3, 4]
    ])

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

    print()
    print(foo.depth)
    foo.step()
    print(foo.depth)
    assert (foo.depth > 0).tolist() == [
        [True, False],
        [False, True]
    ]
