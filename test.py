import unittest
from hough_circle import hough_circle, draw_circle, shrink_image
from detect_extremes import detect_extreme_points
import numpy as np
import time
import pytest
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d


normalize = False
threshold = 6
max_cluster_size = 10
leap_size = 2
shrink_factor = 3

edge_kernel = np.array([[-2, -3, -2], [-3, 20, -3], [-2, -3, -2]]) / 20


def hough_out(img, r):
    edges = np.abs(convolve2d(img, edge_kernel, mode="same", boundary="symm"))
    return hough_circle(edges, radius=r, shrink_factor=shrink_factor)


class TestHoughCircle(unittest.TestCase):
    ### shape
    def test_rectangle_img(self):
        # Image with two circles
        img = np.zeros((200, 100))
        draw_circle(img, 25, 25, 15)

        # Detect circles

        hough_img = hough_out(img, 15)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 25, delta=1)
        self.assertAlmostEqual(circles[0][1], 25, delta=1)

    def test_slender_img(self):
        # Image with two circles
        img = np.zeros((10, 1000))
        draw_circle(img, 5, 5, 2)

        # Detect circles
        hough_img = hough_img = hough_out(img, 2)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 5, delta=1)
        self.assertAlmostEqual(circles[0][1], 5, delta=1)

    ### format
    def RGB_img():
        width = 200
        height = 200

        # Create an empty RGB image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define circle centers and radius
        center1 = (50, 50)  # Top-left corner
        center2 = (150, 150)  # Bottom-right corner
        radius = 10

        # Draw circles in blue and green
        cv2.circle(image, center1, radius, (255, 0, 0), -1)  # Blue circle
        cv2.circle(image, center2, radius, (0, 255, 0), -1)  # Green circle

        hough_img = hough_img = hough_out(image, 10)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 2)
        self.assertAlmostEqual(circles[0][0], 50, delta=1)
        self.assertAlmostEqual(circles[0][1], 50, delta=1)
        self.assertAlmostEqual(circles[1][0], 150, delta=1)
        self.assertAlmostEqual(circles[1][1], 150, delta=1)

    ### object property

    def circle_thick_line():
        width = 200
        height = 200

        # Create an empty RGB image (background color is black)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Circle parameters
        center = (width // 2, height // 2)  # Center at image center
        radius = 15
        thickness = 4  # Line width of the circle

        # Draw circle in white (BGR format)
        cv2.circle(image, center, radius, (255, 255, 255), thickness)

        hough_img = hough_img = hough_out(image, 15)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 100, delta=1)
        self.assertAlmostEqual(circles[0][1], 100, delta=1)

    def test_no_circles():
        img = np.zeros((100, 100))
        circles = detect_extreme_points(
            img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )
        assert len(circles) == 0

    def test_multiple_circles(self):
        # Image with two circles
        img = np.zeros((100, 100))
        draw_circle(img, 25, 25, 15)
        draw_circle(img, 75, 75, 15)

        # Detect circles
        hough_img = hough_out(img, 15)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 2)
        self.assertAlmostEqual(circles[0][0], 25, delta=1)
        self.assertAlmostEqual(circles[0][1], 25, delta=1)
        self.assertAlmostEqual(circles[1][0], 75, delta=1)
        self.assertAlmostEqual(circles[1][1], 75, delta=1)

    # Test bad input
    def test_bad_input():
        with pytest.raises(ValueError):
            detect_extreme_points(
                None,
                threshold=threshold,
                max_cluster_size=max_cluster_size,
                leap_size=leap_size,
                normalize=normalize,
            )

    ### inference
    def create_blurred_image_with_circle(
        width, height, center, radius, blur_kernel_size
    ):
        width = 200
        height = 200
        center = (50, 50)
        radius = 15
        blur_kernel_size = (5, 5)
        # Create an empty RGB image
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Draw a white circle (BGR format)
        cv2.circle(image, center, radius, (255, 255, 255), -1)  # Filled circle

        # Apply Gaussian blur
        image = cv2.GaussianBlur(image, blur_kernel_size, 0)
        hough_img = hough_out(image, 15)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 50, delta=1)
        self.assertAlmostEqual(circles[0][1], 50, delta=1)

    def draw_thick_point():
        width = 200
        height = 200

        # Create an empty RGB image (background color is black)
        image = np.zeros((height, width, 3), dtype=np.uint8)

        # Point coordinates (center of the image)
        center = (width // 2, height // 2)

        # Draw point (considering thickness)
        thickness = -1  # Fills the entire circle (effectively drawing a thick point)
        cv2.circle(image, center, 10, (255, 255, 255), thickness)  # White point
        hough_img = hough_out(image, 10)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )

        # Check if both circles were detected
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 100, delta=1)
        self.assertAlmostEqual(circles[0][1], 100, delta=1)

    ### general

    def test_hough_circle_bad_params(self):
        # Load test image
        img = np.array(Image.open("frog.png").convert("L"))
        edges = np.abs(convolve2d(img, edge_kernel, mode="same", boundary="symm"))
        # Check that it raises an error for bad radius
        with self.assertRaises(ValueError):
            hough_circle(edges, -5)

        # Check that it raises an error for bad shrink factor
        with self.assertRaises(ValueError):
            hough_circle(edges, 40, shrink_factor=0)

    def test_hough_circle_display(self):
        # Load test image
        img = np.array(Image.open("frog.png").convert("L"))
        # Apply hough transform
        result = hough_out(img, 40)

        # Display result
        plt.imshow(result)

        # Check that plot was displayed
        self.assertTrue(plt.fignum_exists(1))

    # Test performance
    def test_performance():
        start = time.time()
        img = np.array(Image.open("frog.png").convert("L"))
        hough_img = hough_out(img, 40)
        circles = detect_extreme_points(
            hough_img,
            threshold=threshold,
            max_cluster_size=max_cluster_size,
            leap_size=leap_size,
            normalize=normalize,
        )
        end = time.time()
        assert (end - start) < 0.5  # complete in under 0.5 sec


if __name__ == "__main__":
    unittest.main()
