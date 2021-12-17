# property based testing with hypothesis -- these take longer so they get their own file

from hypothesis import given, note
from hypothesis.strategies import floats, lists, builds
from src.functions import *
import numpy as np

@given(arr=builds(np.array, lists(floats(min_value=0, max_value=10), min_size=2, max_size=100)))
def test_balancing_preserves_mass(arr):
    diff = abs(np.sum(balance(arr)) - arr[0])
    note(diff)
    assert diff < 1e-6


@given(arr=builds(np.array, lists(floats(min_value=0, max_value=10), min_size=9, max_size=9)))
def test_flatten_unflatten(arr):
    neighbors = get_neighbors(arr.reshape((3, 3)))
    assert (unflatten_unsort(*flatten_sort(neighbors)) == neighbors).all()
