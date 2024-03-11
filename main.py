from hough_circle import hough_circle

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

img = np.array(Image.open("frog.png").convert("L"))
img = hough_circle(img, 40, shrink_factor=3)
plt.imshow(img, cmap="PuBuGn_r")
