import matplotlib.pyplot as plt


def plot_water(water):
    plt.imshow(water, cmap="Blues", vmin=-0.1, vmax=1)
    plt.colorbar()
    plt.show()
