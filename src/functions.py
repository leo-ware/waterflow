import jax
import jax.numpy as jnp


def balance(arr):
    """Balances water among connected bins of specified heights

    Adapted from:
    https://stackoverflow.com/questions/70319574/numpy-balancing-water-levels-between-bins-of-different-depths/
    """
    # unpack inputs
    water = arr[0]
    bin_heights = arr[1:] - jnp.min(arr[1:])

    # useful static
    index = jnp.arange(len(bin_heights))
    zero = jnp.zeros(1)

    depth_diff = jnp.diff(bin_heights)
    volume_for_bottom_bin = jnp.cumsum(jnp.concatenate([zero, depth_diff]) * index)
    idx_last_filled = jnp.searchsorted(volume_for_bottom_bin, water, side="right") - 1
    levels = bin_heights[idx_last_filled] - bin_heights

    remainder = water - volume_for_bottom_bin[idx_last_filled]
    levels = levels + remainder / (idx_last_filled + 1)
    levels = jnp.where(index <= idx_last_filled, levels, 0)

    return levels


# vectorize balance_arr to work on more than one problem at once
v_balance = jax.vmap(balance)


def get_neighbors(grid, pad_mode="edge"):
    """Takes nxn array and returns nxnx3x3 array, where each (i, j) is the moore neighborhood

    Args:
        grid: grid to calculate neighbors for
        pad_mode: 'edge' or 'zero', what value to use for neighbors for spaces at the edge

    Returns:
        neighbors: each (i, j) in `neighbors` contains the moore neighborhood of the corresponding point
            (i, j) in `grid`
    """
    if pad_mode == "edge":
        grid = jnp.pad(grid, 1, mode="edge")
    elif pad_mode == "zero":
        grid = jnp.pad(grid, 1, constant_values=1)
    else:
        raise ValueError("unknown value for parameter pad_edge")

    neighbors = jnp.zeros(grid.shape + (3, 3), dtype=float)
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            thing = jnp.roll(grid, (1 - i, 1 - j), axis=(0, 1))
            neighbors = neighbors.at[:, :, i, j].set(thing)
    return neighbors[1:-1, 1:-1, :, :]


def sum_neighbor_grid(neighbor_grid):
    """Takes a neighbor grid (output of `get_neighbors`) and returns sum across neighbor values

    So, in the neighbor grid, each point (i, j) is a 3x3 array. Each of the elements in this 3x3 array corresponds
    to one of the neighbors of the point. This function sums across all the points in the neighbor grid which
    correspond to each point.

    Args:
        neighbor_grid: nxnx3x3 jax array

    Returns:
        acc: a nxn array
    """
    pad_width = [(1, 1), (1, 1), (0, 0), (0, 0)]
    padded_vmd = jnp.pad(neighbor_grid, pad_width, constant_values=0)

    acc = jnp.zeros(padded_vmd.shape[:2])
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            offset = jnp.roll(padded_vmd, (i, j), (0, 1))[:, :, i + 1, j + 1]
            acc = offset + acc

    acc = acc[1:-1, 1:-1]
    return acc


def flatten_sort(neighbor_grid):
    """Takes a nxnx3x3 grid, reshapes it to (n^2)x9, and sorts the last axis"""
    flat = neighbor_grid.reshape((-1, 9))
    sort_indices = jnp.argsort(flat, axis=1)
    sorted_flat = jnp.take_along_axis(flat, sort_indices, axis=1)
    return sorted_flat, sort_indices


def unflatten_unsort(sorted_flat, sort_indices):
    """Inverse of `flatten_sort`"""
    unsort_indeces = jnp.argsort(sort_indices, axis=-1)
    unsort = jnp.take_along_axis(sorted_flat, unsort_indeces, axis=1)
    n = int(sorted_flat.shape[0]**0.5)
    unflatten = unsort.reshape((n, n, 3, 3))
    return unflatten
