import numpy as np
from scipy.signal import convolve2d
import argparse
from PIL import Image


def get_bilinear_multipliers(x0, y0):
    """
    Get multipliers for bilinear interpolation.

    x0, y0: float ([0; 1]) -- relative coordinates,
            x increasing downward, y increasing to the right
    """
    bl_mult = {"tl": 0.0, "tr": 0.0, "bl": 0.0, "br": 0.0}
    bl_mult["tl"] = (1 - x0) * (1 - y0)
    bl_mult["tr"] = (1 - x0) * y0
    bl_mult["bl"] = x0 * (1 - y0)
    bl_mult["br"] = x0 * y0

    return bl_mult


def get_patch_coords(x, y):
    """
    Determine the top left pixel of a 2x2 pixel group
    used to interpolate given point, and the point's
    coordinates relative to the patch.

    x, y: float -- absolute coordinates,
          x increasing downward, y increasing to the right
    """
    xp = int(np.floor(x - 0.5))
    yp = int(np.floor(y - 0.5))

    x0 = x - xp - 0.5
    y0 = y - yp - 0.5

    return (xp, yp, x0, y0)


def draw_subpixel(canvas, x, y, intensity):
    """
    Place a pixel with non-integer coordinates,
    spreading intensity to neighbouring pixels if they overlap
    """
    xp, yp, x0, y0 = get_patch_coords(x, y)

    bl_mul = get_bilinear_multipliers(x0, y0)

    canvas[xp][yp] += intensity * bl_mul["tl"]
    canvas[xp][yp + 1] += intensity * bl_mul["tr"]
    canvas[xp + 1][yp] += intensity * bl_mul["bl"]
    canvas[xp + 1][yp + 1] += intensity * bl_mul["br"]


def draw_circle(canvas, x, y, radius, point_density=2, npoints=None, intensity=None):
    """
    Draw a circle with given center and radius by incrementing cells on canvas.
    """
    if npoints is None:
        npoints = int(np.ceil(2 * np.pi * radius * point_density))
    if intensity is None:
        intensity = 1 / npoints

    for i in range(npoints):
        x0 = x - radius * np.sin(2 * np.pi * i / npoints)
        y0 = y + radius * np.cos(2 * np.pi * i / npoints)

        if (
            x0 < 0.5
            or y0 < 0.5
            or x0 >= len(canvas) - 0.5
            or y0 >= len(canvas[0]) - 0.5
        ):
            continue
        draw_subpixel(canvas, x0, y0, intensity)


def get_circle_kernel(radius, point_density=2, npoints=None, intensity=None):
    k_size = int(np.ceil(radius * 2 + 1))
    kernel = np.zeros((k_size, k_size))
    draw_circle(
        kernel,
        k_size / 2,
        k_size / 2,
        radius,
        point_density=point_density,
        npoints=npoints,
        intensity=intensity,
    )
    return kernel


def shrink_image(image, factor):
    """
    Naive image scaling to compute results faster; produces less artifacts with factor > 2
    """
    scaled_height = int(np.ceil(image.shape[0] / factor + 1))
    scaled_width = int(np.ceil(image.shape[1] / factor + 1))
    image_shrunk = np.zeros((scaled_height, scaled_width))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            draw_subpixel(
                image_shrunk,
                0.5 + x / factor,
                0.5 + y / factor,
                image[x, y] / factor**2,
            )
    return image_shrunk


def hough_circle(
    source_img, radius, point_density=2.0, shrink_factor=1.0, conv_mode="same"
):
    if shrink_factor > 1.0:
        source_img = shrink_image(source_img, shrink_factor)
        radius /= shrink_factor
    kernel = get_circle_kernel(radius, point_density=point_density)

    return convolve2d(source_img, kernel, mode=conv_mode)
