from src.functions import *
import jax
import jax.numpy as jnp
import numpy as np


class Water:
    def __init__(self, ground_level, water_level=None, chance_rain=None):
        self.ground_level = jnp.array(ground_level)
        self.water_level = jnp.array(water_level) if water_level is not None else jnp.zeros_like(self.ground_level)
        self.chance_rain = jnp.array(chance_rain) if chance_rain is not None else jnp.zeros_like(self.ground_level)

    # @jax.jit
    def gravity(self):
        """Calculates new water layout based on effects of gravity"""
        neighbors = get_neighbors(self.ground_level + self.water_level)
        neighbors = neighbors.at[:, :, 1, 1].set(self.ground_level)
        flat, sort_indices = flatten_sort(neighbors)
        water_flat = self.water_level.reshape((-1, 1))

        stacked = jnp.hstack([water_flat, flat])
        balanced = v_balance(stacked)
        unsorted_balanced = unflatten_unsort(balanced, sort_indices)

        new_water = sum_neighbor_grid(unsorted_balanced)
        return new_water

    def rainfall(self):
        """Calculates water to add from rainfall"""
        rain = (np.random.random(self.chance_rain.shape) <= self.chance_rain).astype(int)
        return rain

    def step(self):
        self.water_level = self.gravity() + self.rainfall()
