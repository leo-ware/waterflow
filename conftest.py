import numpy as np

dem = np.array([
    [3, 3, 3, 3],
    [2, 2, 2, 2],
    [1, 1, 1, 1],
    [0, 0, 0, 0]
], dtype=float)

source = np.array([
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
])

manning = np.full_like(dem, 0.02)

