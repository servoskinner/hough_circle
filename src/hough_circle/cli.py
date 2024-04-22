from hough_circle import hough_circle, sobel
from detect_extremes import detect_extreme_points, to_bit_array

import numpy as np
import argparse
import json

from PIL import Image
from scipy.signal import convolve2d

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect circles on an image using generalized Hough transform."
    )

    # required parameters
    parser.add_argument(
        "-i", "--img", "--image", type=str, help="Path to target image."
    )
    parser.add_argument(
        "-r", "--rad", "--radius", type=float, help="Radius of circles to find"
    )
    parser.add_argument(
        "-t", "--thr", "--thres", "--threshold", type=float, help="Threshold for "
    )
    # optional parameters
    parser.add_argument(
        "-l",
        "--leap",
        "--leap-size",
        type=int,
        default=1,
        help="Leap size for cluster traversal",
    )
    parser.add_argument(
        "-c",
        "--cluster",
        "--cluster-size",
        type=int,
        default=1,
        help="Max cluster diameter",
    )
    parser.add_argument(
        "-s",
        "--shrink",
        "--shrink-factor",
        type=int,
        default=1,
        help="Shrink factor: greater values speed up processing at expense of precision",
    )
    # boolean flags
    parser.add_argument(
        "-e",
        "--edge",
        "--edges",
        action="store_true",
        help="Apply edge filter before detecting circles",
    )
    parser.add_argument(
        "--small-kernel", action="store_true", help="Use 2x2 kernel for edge detection"
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        help="Normalize transformed image before binarization",
    )
    # output control
    parser.add_argument(
        "-o",
        "--out",
        "--output",
        type=str,
        default="out",
        help=".json file to write results to",
    )
    parser.add_argument(
        "--raw",
        type=str,
        default=None,
        help="Save transformed image with specified name",
    )

    args = vars(parser.parse_args())

    img = np.array(Image.open(args["img"]).convert("L"))
    if args["edge"]:
        img = sobel(img, small_kernel=args["small_kernel"])

    print("Applying Hough transform...")
    hough = hough_circle(img, args["rad"], shrink_factor=args["shrink"])

    if args["raw"] is not None:
        Image.fromarray((hough / np.max(hough) * 255).astype("uint8")).save(args["raw"])
        print("Raw transformed image has been saved")

    print("Detecting extreme points...")
    detections = detect_extreme_points(
        hough,
        threshold=args["thr"],
        max_cluster_size=args["cluster"],
        leap_size=args["leap"],
        normalize=args["normalize"],
        shrink_factor=args["shrink"],
    )

    output = {
        "x": list(item[0] for item in detections),
        "y": list(item[1] for item in detections),
    }

    with open(
        args["out"] if args["out"].endswith(".json") else (args["out"] + ".json"), "w"
    ) as outfile:
        json.dump(output, outfile)

    print("Result has been written and saved")
