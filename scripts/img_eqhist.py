#!/usr/bin/env python3


"""
Apply histogram equalization to the given images.
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
    "--mode",
    type=str,
    choices=["global", "clahe"],
    default="clahe",
    help="Mode of histogram equalization - simple global histogram equalization or CLAHE (Contrast Limited Adaptive Histogram Equalization)"
)
parser.add_argument(
    "--clahe_contrast_limit",
    type=float,
    default=2.0,
    help="CLAHE contrast limit value",
)


def main(args):
    if os.path.isfile(args.input_path):
        img = cv2.imread(args.input_path)
        img_eq = equalize_img(img, args.mode, args.clahe_contrast_limit)
        cv2.imwrite(args.output_path, img_eq)
    else:
        os.makedirs(args.output_path, exist_ok=True)
        for root, _, files in os.walk(args.input_path):
            for file in files:
                if file.endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(root, file))
                    img_eq = equalize_img(img, args.mode, args.clahe_contrast_limit)
                    cv2.imwrite(os.path.join(args.output_path, file), img_eq)


def equalize_img(img, mode, clahe_contrast_limit=2.0):
    """Equalize the histogram of the image.

    Parameters:
    img (np.ndarray): The input image.
    mode (str): Mode of histogram equalization - simple global histogram 
        equalization or CLAHE

    Returns:
    img_eq (np.ndarray): The image with equalized histogram.

    """

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the luma component

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clahe_contrast_limit, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    elif mode == "global":
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

    return img_eq


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)