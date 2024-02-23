#!/usr/bin/python3


"""
Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.

"""


import os
import argparse
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("img1", type=str, help="Path to the first image")
parser.add_argument("img2", type=str, help="Path to the second image")


def main(args):
    assert os.path.exists(args.img1)
    assert os.path.exists(args.img2)

    img1 = np.asarray(Image.open(args.img1))
    img2 = np.asarray(Image.open(args.img2))

    assert img1.shape == img2.shape
    w, h, c = img1.shape

    mse = np.sum(np.square(img1 - img2), axis=None) / (w * h * c)
    psnr = 10 * np.log10(255**2 / mse)

    print(psnr)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
