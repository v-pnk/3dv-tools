#!/usr/bin/env python3


"""
Apply gamma correction on the given images.
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
    "--gamma",
    type=float,
    default=1.0,
    help="Gamma value (> 0)."
)


def main(args):
    if args.gamma <= 0:
        raise ValueError("Gamma value should be greater than 0.")

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

        print(img.dtype)

        if img.dtype == "uint8":
            img_new = 255.0 * ((img / 255.0) ** args.gamma)
        elif img.dtype == "uint16":
            img_new = 65535.0 * ((img / 65535.0) ** args.gamma)
        elif img.dtype == "float32":
            img_new = img ** args.gamma

        cv2.imwrite(file_out, img_new)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)