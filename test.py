import unittest
from hough_circle import hough_circle, draw_circle, shrink_image
import numpy as np


class TestHoughCircle(unittest.TestCase):

    def test_circle_detection(self):
        # Generate test image with known circles
        img = np.zeros((100, 100))
        draw_circle(img, 50, 50, 20)
        draw_circle(img, 30, 30, 10)

        # Detect circles
        circles = hough_circle(img, radius=21)

        # Check if expected circles were found
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0][0], 50, delta=1)
        self.assertAlmostEqual(circles[0][1], 50, delta=1)
        self.assertAlmostEqual(circles[0][2], 20, delta=1)

    def test_multiple_circles(self):
        # Image with two circles
        img = np.zeros((100, 100))
        draw_circle(img, 25, 25, 15)
        draw_circle(img, 75, 75, 15)

        # Detect circles
        circles = hough_circle(img, radius=15)

        # Check if both circles were detected
        self.assertEqual(len(circles), 2)
        self.assertAlmostEqual(circles[0][0], 25, delta=1)
        self.assertAlmostEqual(circles[0][1], 25, delta=1)
        self.assertAlmostEqual(circles[1][0], 75, delta=1)
        self.assertAlmostEqual(circles[1][1], 75, delta=1)

    def test_shrink_image(self):
        # Generate test image
        img = np.zeros((100, 100))
        draw_circle(img, 50, 50, 20)

        # Shrink image
        img_small = shrink_image(img, 2)

        # Check scaled size
        self.assertEqual(img_small.shape[0], 50)
        self.assertEqual(img_small.shape[1], 50)

        # Check if circle was properly scaled
        circles = hough_circle(img_small, radius=10)
        self.assertAlmostEqual(circles[0][0], 25, delta=1)
        self.assertAlmostEqual(circles[0][1], 25, delta=1)


######################
import unittest
from hough_circle import hough_circle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Test on image with known circles
@pytest.mark.parametrize(
    "image_file, expected_circles",
    [
        ("circles.png", [(50, 50, 20), (100, 150, 30)]),
        ("circles_large.png", [(100, 120, 50), (200, 250, 80)]),
    ],
)
class TestHoughCircle(unittest.TestCase):
    # Test no circles
    def test_no_circles():
        circles = detect_circles("blank.png")
        assert len(circles) == 0

    def test_hough_circle_on_simple_image(self):
        # Create a simple test image with a known circle
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)

        # Apply hough transform
        result = hough_circle(img, 30)

        # Check that the result contains a circle at the expected location
        assert (50, 50) in result.circles

        # Optionally check number of circles detected
        assert len(result.circles) == 1

    def test_detect_known_circles(image_file, expected_circles):
        circles = detect_circles(image_file)
        assert len(circles) == len(expected_circles)
        assert all([c in expected_circles for c in circles])

    def test_hough_circle_on_complex_image(self):
        # Load a test image with multiple circles
        # TODO image "circles.png"

        test_img = np.array(Image.open("circles.png").convert("L"))

        # Apply hough transform
        result = hough_circle(test_img)

        # Check that the expected number of circles are detected
        # Here we assume the test image contains 3 circles
        assert len(result.circles) == 3

    def test_hough_circle_bad_params(self):
        # Load test image
        img = np.array(Image.open("frog.png").convert("L"))

        # Check that it raises an error for bad radius
        with self.assertRaises(ValueError):
            hough_circle(img, -5)

        # Check that it raises an error for bad shrink factor
        with self.assertRaises(ValueError):
            hough_circle(img, 30, shrink_factor=0)


def test_hough_circle_display(self):
    # Load test image
    img = np.array(Image.open("frog.png").convert("L"))

    # Apply hough transform
    result = hough_circle(img, 30)

    # Display result
    plt.imshow(result)

    # Check that plot was displayed
    self.assertTrue(plt.fignum_exists(1))


# Test bad input
def test_bad_input():
    with pytest.raises(ValueError):
        detect_circles(None)


# Test performance
def test_performance():
    start = time.time()
    detect_circles("circles.png")
    end = time.time()
    assert (end - start) < 0.5  # complete in under 0.5 sec


if __name__ == "__main__":
    unittest.main()


###################3


# Test different parameter values
@pytest.mark.parametrize("param, min_expected, max_expected", [(50, 1, 3), (100, 1, 5)])
def test_param_threshold(param, min_expected, max_expected):
    circles = detect_circles("circles.png", param=param)
    assert len(circles) >= min_expected
    assert len(circles) <= max_expected
