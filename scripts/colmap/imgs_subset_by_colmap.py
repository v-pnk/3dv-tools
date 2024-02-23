#!/usr/bin/env python3


"""
Copy a subset of images from a directory to another directory based on the list 
of images in a COLMAP model.
"""


import os
import pycolmap
import argparse
import shutil


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "input_img_dir", 
    type=str, 
    help="Input directory with images"
)
parser.add_argument(
    "colmap", 
    type=str, 
    help="COLMAP model containing subset of images"
)
parser.add_argument(
    "output_img_dir", 
    type=str, 
    help="Output directory for the subset of images"
)
parser.add_argument(
    "--ignore_ext",
    action="store_true",
    help="Ignore extensions when comparing image names",
)
parser.add_argument(
    "--invert",
    action="store_true",
    help="Invert the selection, i.e. copy images that are not in the model",
)


def main(args):
    assert os.path.isdir(args.input_img_dir)

    model = pycolmap.Reconstruction(args.colmap)
    # input_images = os.listdir(args.input_img_dir)
    input_images = [
        os.path.relpath(os.path.join(dp, f), args.input_img_dir)
        for dp, _, filenames in os.walk(args.input_img_dir)
        for f in filenames
    ]

    img_list = [img.name for img in model.images.values()]

    if args.ignore_ext:
        img_list = [os.path.splitext(img)[0] for img in img_list]

    if args.invert:
        output_images = [img for img in input_images if img not in img_list]
        if args.ignore_ext:
            output_images = [
                img for img in input_images if os.path.splitext(img)[0] not in img_list
            ]
    else:
        output_images = [img for img in input_images if img in img_list]
        if args.ignore_ext:
            output_images = [
                img for img in input_images if os.path.splitext(img)[0] in img_list
            ]

    input_paths = [os.path.join(args.input_img_dir, img) for img in output_images]
    output_paths = [os.path.join(args.output_img_dir, img) for img in output_images]

    for i in range(len(input_paths)):
        os.makedirs(os.path.dirname(output_paths[i]), exist_ok=True)
        shutil.copyfile(input_paths[i], output_paths[i])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
