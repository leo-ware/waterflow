from src.water import Water
import numpy as np
from pytest import approx


def test_gravity():
    foo = Water(
        ground=np.zeros((3, 3)),
        water=[
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
    )
    spread_out = np.ones((3, 3))/9

    assert (foo.gravity().flatten()) == approx(list(spread_out.flatten()))
