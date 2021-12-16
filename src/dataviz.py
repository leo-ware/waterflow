import matplotlib.pyplot as plt
import numpy as np
from src.water import Water


def plot_water_ax(water_level, ax, vmax=None):
    if vmax is None:
        vmax = np.max(water_level)

    ax.imshow(np.array(water_level), cmap="Blues", vmin=-0.1, vmax=vmax)
    # ax.colorbar()
    ax.set_xticks([])
    ax.set_yticks([])


def plot_water_height(water: Water, vmax=10):
    fig, (left, right) = plt.subplots(1, 2)

    for ax in (left, right):
        ax.set_xticks([])
        ax.set_yticks([])

    left.set_title("Water Level")
    plot_water_ax(water.water_level, left, vmax)

    right.set_title("Total Height")
    right.imshow(water.total_height, cmap="Browns", vmin=0, vmax=np.max(water.total_height))
    right.colorbar()

    return fig


def plot_time_slices(water: Water, times):
    # collect water history
    times = set(times)
    water_history = {}
    for _ in range(int(1e6)):
        if water.t in times:
            times.remove(water.t)
            water_history[water.t] = np.array(water.water_level)

        water.step()
        if not times:
            break

    # plot stuff
    fig, axes = plt.subplots(1, len(water_history), figsize=(8, 4))
    vmax = np.max(list(water_history.values()))
    for ax, (t, depth) in zip(axes, water_history.items()):
        plot_water_ax(depth, ax, vmax)
        ax.set_title(f"step={t}")

    return fig
