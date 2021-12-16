from src.functions import *
import jax
import jax.numpy as jnp


class Water:
    def __init__(self, ground, water=None):
        self.ground = jnp.array(ground)
        if water is not None:
            self.water = jnp.array(water)
        else:
            self.water = jnp.zeros_like(water)

    # @jax.jit
    def gravity(self):
        """Calculates new water layout based on effects of gravity"""
        neighbors = get_neighbors(self.ground + self.water)
        neighbors = neighbors.at[:, :, 1, 1].set(self.ground)
        flat, sort_indices = flatten_sort(neighbors)
        water_flat = self.water.reshape((-1, 1))

        stacked = jnp.hstack([water_flat, flat])
        balanced = v_balance(stacked)
        unsorted_balanced = unflatten_unsort(balanced, sort_indices)

        new_water = sum_neighbor_grid(unsorted_balanced)
        return new_water

    def step(self):
        self.water = self.gravity()
