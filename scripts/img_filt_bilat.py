#!/usr/bin/env python3


"""
Apply bilateral filter on the images.
"""


import os
import argparse

import cv2


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_path", 
    type=str, 
    help="Path to the input image or directory"
)
parser.add_argument(
    "output_path", 
    type=str, 
    help="Path to the output image or directory"
)
parser.add_argument(
    "--neigh_d",
    type=int,
    default=9,
    help="Diameter of each pixel neighborhood that is used during filtering."
)
parser.add_argument(
    "--sigma_color",
    type=int,
    default=75,
    help="Filter sigma in the color space."
)
parser.add_argument(
    "--sigma_space",
    type=int,
    default=75,
    help="Filter sigma in the coordinate space."
)


def main(args):
    input_files = []
    output_files = []
    if os.path.isfile(args.input_path):
        input_files.append(args.input_path)
        if os.path.isdir(args.output_path):
            output_files.append(os.path.join(args.output_path, os.path.basename(args.input_path)))
        else:
            output_files.append(args.output_path)
    else:
        os.makedirs(args.output_path, exist_ok=True)
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    input_files.append(os.path.join(root, file))
                    output_files.append(os.path.join(args.output_path, file))

    for file_in, file_out in zip(input_files, output_files):
        img = cv2.imread(file_in)
        img_filt = cv2.bilateralFilter(img, args.neigh_d, args.sigma_color, args.sigma_space)
        cv2.imwrite(file_out, img_filt)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)