import numpy as np
from pytest import raises
from src.water import Water, balance_levels, sum_neighbor_grid, calc_at_neighbors


def test_init():
    arr = [
        [1, 2],
        [3, 4]
    ]
    foo = Water(arr)
    assert foo.dem.tolist() == arr


def test_errs():
    with raises(ValueError):
        Water(np.array([]))

    with raises(ValueError):
        Water(np.array([[]]))


def test_balance_levels():
    assert balance_levels(np.array([2, 1, 3]), 0).tolist() == [0, 0, 0]
    assert balance_levels(np.array([2, 1, 3]), 0.1).tolist() == [0, 0.1, 0]
    assert balance_levels(np.array([2, 1, 3]), 2).tolist() == [0.5, 1.5, 0]
    assert balance_levels(np.array([2, 1, 3]), 3).tolist() == [1, 2, 0]
    assert balance_levels(np.array([2, 1, 3]), 5).tolist() == [1 + 2/3, 2 + 2/3, 2/3]


def test_calc_at_neighbors():
    foo = calc_at_neighbors(np.array([
        [1, 2],
        [3, 4]
    ]))

    assert foo[0, 0].tolist() == [
        [1, 1, 2],
        [1, 1, 2],
        [3, 3, 4]
    ]


def test_sum_neighbor_grid():
    grid = np.array([
        [1, 2],
        [3, 4],
    ])
    assert sum_neighbor_grid(calc_at_neighbors(grid)).tolist() == (4*grid).tolist()


def test_calc_outflow():
    foo = Water(np.zeros((3, 3)))
    foo.depth = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])
    assert foo.calc_outflow().tolist() == [
        [0, 0, 0],
        [0, 8/9, 0],
        [0, 0, 0]
    ]

    foo.depth = np.array([
        [0, 0, 0],
        [0, 1, 0],
        [0, 1, 0]
    ])
    assert foo.calc_outflow().tolist() == [
        [0.0, 0.0, 0.0],
        [0.0, 7/9, 0.0],
        [0.0, 0.6666666666666667, 0.0]
    ]


def test_calc_inflow():
    foo = Water([
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ])

    outflow = foo.calc_outflow()
    inflow = foo.calc_inflow(outflow)

    assert False


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
