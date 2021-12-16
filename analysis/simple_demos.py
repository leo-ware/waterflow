import numpy as np
import matplotlib.pyplot as plt

from src.water import Water
from src.dataviz import plot_time_slices


def plot_one_drop_spread():
    ground = np.zeros((5, 5))
    water = np.zeros((5, 5))
    water[2, 2] = 1
    sim = Water(ground_level=ground, water_level=water)

    fig = plot_time_slices(sim, [0, 1, 2, 3, 4])
    fig.suptitle("Simple Runoff Demo")
    fig.savefig("imgs/one_drop_spread.png")
    plt.show()


if __name__ == "__main__":
    plot_one_drop_spread()
