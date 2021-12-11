import numpy as np


class Water:
    def __init__(self, dem, source=None, manning=0.02):
        self.dem = np.array(dem, dtype=float)

        if len(self.dem.shape) != 2 or not all(self.dem.shape):
            raise ValueError("dem must be 2-dimensional")

        self.source = source if source else np.zeros_like(self.dem)
        self.manning = manning if type(manning) == np.ndarray else np.full_like(self.dem, manning)

        self.depth = np.zeros_like(self.dem)
        self.slope = self.calc_slope()
        self.direction = self.calc_flow_direction()
        self.volume_moving = self.calc_volume_moving()

    def calc_flow_direction(self):
        # extend the dem with a border that is level with it
        padded_dem = np.pad(self.dem, 1, mode="edge")

        neighbors = np.zeros(padded_dem.shape + (3, 3))
        for i in [0, 1, 2]:
            for j in [0, 1, 2]:
                neighbors[:, :, i, j] = np.roll(padded_dem, (1 - i, 1 - j), axis=(0, 1))
        neighbors[:, :, 1, 1] = np.inf

        mins = np.min(neighbors, axis=(2, 3))
        direction = mins.reshape(mins.shape + (1, 1)) == neighbors
        n_directions = np.sum(direction, axis=(2, 3)).reshape(direction.shape[:2] + (1, 1))
        direction = direction/n_directions
        return direction[1:-1, 1:-1]

    def calc_slope(self):
        dx, dy = np.gradient(self.dem)
        return (dx**2 + dy**2)**0.5

    def calc_volume_moving(self):
        # dimensions??
        water_speed = self.depth**(3/2)*self.slope**(1/2)/self.manning
        flooded = water_speed > 1
        volume_moving = water_speed*self.depth
        volume_moving[~flooded] = 0
        return volume_moving

    def calc_inflow(self, volume_moving):
        volume_moving_directional = self.direction * volume_moving.reshape(volume_moving.shape + (1, 1))
        npad = [(1, 1), (1, 1), (0, 0), (0, 0)]
        padded_vmd = np.pad(volume_moving_directional, npad, constant_values=0)

        acc = np.zeros(padded_vmd.shape[:2])
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                acc += np.roll(padded_vmd, (i, j), (0, 1))[:, :, i + 1, j + 1]

        return acc[1:-1, 1:-1]

    def calc_inundation(self):
        return np.zeros_like(self.dem)

    def calc_update(self):
        volume_moving = self.calc_volume_moving()

        outflow = -volume_moving
        inflow = self.calc_inflow(volume_moving)
        inundation = self.calc_inundation()

        return self.source + outflow + inflow + inundation

    def step(self):
        self.depth += self.calc_update()
