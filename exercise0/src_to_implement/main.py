import numpy as np
import matplotlib.pyplot as plt

from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


if __name__ == '__main__':
    # Checkerboard
    c = Checker(250, 25)
    c.draw()
    c.show()

    # Circle
    circ = Circle(1024, 200, (512, 256))
    circ.draw()
    circ.show()

    # Spectrum
    s = Spectrum(255)
    s.draw()
    s.show()

    # Image Generator
    gen = ImageGenerator('./exercise_data/', './Labels.json', 9, [32, 32, 3],
                         rotation=True, mirroring=True, shuffle=True)
    gen.show()