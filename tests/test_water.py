from src.water import Water
import numpy as np
from pytest import approx


def test_gravity():
    foo = Water(
        ground_level=np.zeros((3, 3)),
        water_level=[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
    )
    spread_out = np.ones((3, 3))/9

    assert (foo.gravity().flatten()) == approx(list(spread_out.flatten()))


def test_rainfall():
    shape = (100, 100)
    n_cells = np.prod(shape)

    foo = Water(ground_level=np.zeros(shape), chance_rain=np.zeros(shape))
    assert np.sum(foo.rainfall()) == 0

    foo = Water(ground_level=np.zeros(shape), chance_rain=np.ones(shape))
    assert np.sum(foo.rainfall()) == n_cells

    foo = Water(ground_level=np.zeros(shape), chance_rain=np.full(shape, 0.8))
    assert 0 < np.sum(foo.rainfall()) < n_cells
