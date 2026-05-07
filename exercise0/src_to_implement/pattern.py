import numpy as np
import matplotlib.pyplot as plt


class Checker:
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        # build one 2x2 tile block, then tile it
        tile = np.zeros((tile_size := self.tile_size, tile_size), dtype=float)
        block = np.concatenate([
            np.concatenate([np.zeros((tile_size, tile_size)), np.ones((tile_size, tile_size))], axis=1),
            np.concatenate([np.ones((tile_size, tile_size)), np.zeros((tile_size, tile_size))], axis=1)
        ], axis=0)
        reps = self.resolution // (2 * self.tile_size)
        self.output = np.tile(block, (reps, reps))
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Circle:
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position  # (x, y) = (col, row)
        self.output = None

    def draw(self):
        # position is (x, y): x = column, y = row
        x_center, y_center = self.position
        # meshgrid: x varies along columns, y varies along rows
        x = np.arange(self.resolution)
        y = np.arange(self.resolution)
        xx, yy = np.meshgrid(x, y)
        self.output = ((xx - x_center) ** 2 + (yy - y_center) ** 2) < self.radius ** 2
        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap='gray')
        plt.show()


class Spectrum:
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        n = self.resolution
        # x goes 0->1 left to right, y goes 0->1 top to bottom
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xx, yy = np.meshgrid(x, y)  # xx varies along cols, yy along rows

        # From the corner colors in the figure:
        # top-left: blue (0,0,1), top-right: red (1,0,0)
        # bottom-left: cyan (0,1,1), bottom-right: yellow (1,1,0)
        # R channel: increases left->right (xx)
        # G channel: increases top->bottom (yy)
        # B channel: decreases left->right (1-xx)
        r = xx
        g = yy
        b = 1.0 - xx

        self.output = np.stack([r, g, b], axis=2)
        return self.output.copy()

    def show(self):
        plt.imshow(self.output)
        plt.show()