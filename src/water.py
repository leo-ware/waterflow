import numpy as np


def sum_neighbor_grid(neighbor_grid):
    pad_width = [(1, 1), (1, 1), (0, 0), (0, 0)]
    padded_vmd = np.pad(neighbor_grid, pad_width, constant_values=0)

    acc = np.full(padded_vmd.shape[:2], 0)
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            offset = np.roll(padded_vmd, (i, j), (0, 1))[:, :, i + 1, j + 1]
            acc = offset + acc

    return acc[1:-1, 1:-1]


def get_neighbors(grid):
    neighbors = np.zeros(grid.shape + (3, 3))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            neighbors[:, :, i, j] = np.roll(grid, (1 - i, 1 - j), axis=(0, 1))
    return neighbors[1:-1, 1:-1, :, :]


class Water:
    def __init__(self, dem, source=None):
        self.dem = np.array(dem, dtype=float)
        self.source = source if source else np.zeros_like(self.dem, dtype=float)
        self.depth = np.zeros_like(self.dem, dtype=float)

        if len(self.dem.shape) != 2 or not all(self.dem.shape):
            raise ValueError("dem must be 2-dimensional")

    def calc_height(self):
        padded_dem = np.pad(self.dem, 1, mode="edge")
        padded_depth = np.pad(self.depth, 1, constant_values=0)
        padded_height = padded_depth + padded_dem
        return get_neighbors(padded_height)

    def calc_outflow(self, height):
        lower = height <= height[:, :, 1, 1].reshape(height.shape[:2] + (1, 1))

        # find the average height among lower cells
        n_lower = np.sum(lower, axis=(2, 3))
        lower_sum = np.sum(np.where(lower, height, 0), axis=(2, 3))
        lower_avg = lower_sum / n_lower

        # remove water to bring lower neighboring cells up to the average height
        return np.minimum(self.dem + self.depth - lower_avg, self.depth)

    def calc_inflow(self, outflow, height):
        height = height.copy()
        height[:, :, 1, 1] -= outflow
        height = height.reshape(self.dem.shape + (-1,))

        sort_indices = np.argsort(height)
        sorted_heights = np.take_along_axis(height, sort_indices, -1)

        # whole-level water allocations
        volume_needed_to_fill = np.cumsum(
            np.diff(sorted_heights, axis=-1) *
            np.arange(1, 9).reshape((1, 1, -1)),
            axis=-1)
        index_last_filled = np.sum(volume_needed_to_fill <= outflow.reshape(3, 3, 1), axis=-1).reshape([3, 3, 1])
        fill_to_depth = np.take_along_axis(sorted_heights, index_last_filled, -1)
        relative_heights = sorted_heights - np.min(sorted_heights, axis=-1).reshape([3, 3, 1])
        integer_allocations = fill_to_depth - relative_heights

        # fractional-level water allocations
        used_water = np.take_along_axis(volume_needed_to_fill, index_last_filled - 1, -1)
        used_water[index_last_filled == 0] = 0
        remainder = outflow - used_water.reshape([3, 3])
        fractional_allocations = remainder.reshape(3, 3, 1) / index_last_filled

        # combine and zero out non-allocating spaces
        allocations = fractional_allocations + integer_allocations
        nonallocating = np.arange(9).reshape([1, 1, 9]) > index_last_filled
        allocations[nonallocating] = 0

        # rearrange results
        inverse_sort_indices = np.take_along_axis(np.arange(9).reshape([1, 1, 9]), sort_indices, -1)
        unsorted_allocations = np.take_along_axis(allocations, inverse_sort_indices, -1)
        final_allocations = sum_neighbor_grid(unsorted_allocations.reshape(self.dem.shape + (3, 3)))

        return final_allocations

    def calc_update(self):
        height = self.calc_height()
        outflow = self.calc_outflow(height)
        inflow = self.calc_inflow(outflow, height)
        return self.source + inflow

    def step(self):
        self.depth = self.calc_update()
