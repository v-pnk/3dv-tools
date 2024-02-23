#!/usr/bin/env python3


"""
Evaluate blurriness of given images by using the average value of local standard
deviation.

"""


import argparse
import os
import cv2
import numpy as np
import shutil
import glob


parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument(
    "input_image", 
    type=str, 
    help="path to the input image or directory"
)
parser.add_argument(
    "--kernel_size", 
    type=int, 
    default=3, 
    help="size of the blurring kernel"
)
parser.add_argument(
    "--threshold",
    type=float,
    default=6,
    help="the threshold of the max. of local std. dev. to label the image as blurry (sharper images have higher local std. dev.)",
)
parser.add_argument(
    "--delete_blurry", 
    action="store_true", 
    help="delete the blurry images"
)
parser.add_argument(
    "--copy_to", 
    type=str, 
    help="copy the sharp images to the given directory"
)
parser.add_argument(
    "--test_postfix",
    type=str,
    default="",
    help="only the images with the given postfix will be tested (leave blank to test all)",
)
parser.add_argument(
    "--process_postfix",
    type=str,
    default="",
    help="only the images with the given postfix will be deleted or copied (leave blank to process all)",
)


# - code taken from https://stackoverflow.com/questions/11456565/opencv-mean-sd-filter
def main(args):
    input_files = []

    assert os.path.exists(args.input_image)

    if os.path.isfile(args.input_image):
        input_files.append(args.input_image)
    else:
        dirfiles = os.listdir(args.input_image)
        input_files = [
            os.path.join(args.input_image, f)
            for f in dirfiles
            if os.path.isfile(os.path.join(args.input_image, f))
        ]

    input_files.sort()

    for img_file in input_files:
        assert os.path.exists(img_file)

        if not (img_file.endswith(args.test_postfix)):
            continue

        img = cv2.imread(img_file).astype(np.float32)
        mu = cv2.blur(img, (args.kernel_size, args.kernel_size))
        mu2 = cv2.blur(img * img, (args.kernel_size, args.kernel_size))
        sigma = cv2.sqrt(np.abs(mu2 - mu * mu))
        sigma = np.sum(sigma, axis=2) / 3
        avg_sd = np.mean(sigma)

        process_files = list()
        if (args.copy_to is not None or args.delete_blurry) and not (
            args.process_postfix
        ):
            process_files = [
                file for file in glob.glob(img_file[: -len(args.test_postfix)] + "*")
            ]
        else:
            process_files = [img_file]

        if avg_sd < args.threshold:
            label = "blurry"

            if args.delete_blurry:
                for proc_file in process_files:
                    os.remove(proc_file)
        else:
            label = "sharp"

            if args.copy_to is not None:
                for proc_file in process_files:
                    file_name = os.path.basename(proc_file)
                    new_file_path = os.path.join(args.copy_to, file_name)
                    shutil.copyfile(proc_file, new_file_path)

        print("{}\t\t{:.2f}\t\t{}".format(os.path.basename(img_file), avg_sd, label))


def mat2gray(mat):
    dst = np.zeros(mat.shape[0:2])
    dst = cv2.normalize(
        src=mat, dst=dst, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX
    )
    return dst


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
