from src.functions import *
import jax
import jax.numpy as jnp
import numpy as np


class Water:
    def __init__(self, ground_level, water_level=None, chance_rain=None, inundation_max=0.04, inundation_rate=0.2, rain_amount=0.01):
        """Class to simulate rainfall, inundation, and runoff mechanics

        Relies on a number of helper functions in src/functions.py

        Args:
            ground_level: nxn grid with height of the terrain (in meters)
            water_level: nxn grid with depth of water (in meters)
            chance_rain: nxn grid with the percentage chance of rain occurring
            rain_amount: the amount of water that is distributed if rain occurs at a space
            inundation_max: maximum soil saturation for each space
            inundation_rate: rate of soil saturation (portion of remaining saturation which can be absorbed per step)
        """

        self.ground_level = jnp.array(ground_level)
        self.water_level = jnp.array(water_level) if water_level is not None else jnp.zeros_like(self.ground_level)
        self.chance_rain = jnp.array(chance_rain) if chance_rain is not None else jnp.zeros_like(self.ground_level)
        self.water_saturation = jnp.zeros_like(self.ground_level)

        self.inundation_max = inundation_max
        self.inundation_rate = inundation_rate
        self.rain_amount = rain_amount

        self.t = 0

    def rainfall(self):
        """Calculates water to add from rainfall"""
        rain = (np.random.random(self.chance_rain.shape) <= self.chance_rain).astype(int)
        return rain

    def inundation(self):
        """Calculates the water lost to inundation"""
        inundation_update = (self.inundation_max - self.water_saturation)*self.inundation_rate
        return inundation_update

    # @jax.jit
    def gravity(self):
        """Calculates new water layout based on effects of gravity"""
        # neighbord grid of total height at each space
        neighbors = (get_neighbors(self.ground_level, pad_mode="edge") +
                     get_neighbors(self.water_level, pad_mode="zero"))

        # remove excess water
        neighbors = neighbors.at[:, :, 1, 1].set(self.ground_level)

        flat, sort_indices = flatten_sort(neighbors)
        water_flat = self.water_level.reshape((-1, 1))

        # balance water among neighborhood
        stacked = jnp.hstack([water_flat, flat])
        balanced = v_balance(stacked)
        unsorted_balanced = unflatten_unsort(balanced, sort_indices)

        new_water = sum_neighbor_grid(unsorted_balanced)
        return new_water

    @property
    def total_height(self):
        """total_height = ground_level + water_level"""
        return self.water_level + self.ground_level

    def step(self):
        """Update function for the model"""
        self.t += 1

        self.water_level = self.water_level + self.rainfall()
        inundation = self.inundation()
        self.water_saturation = self.water_saturation + inundation
        self.water_level = self.water_level - inundation
        self.water_level = self.water_level + self.gravity()
