#!/usr/bin/env python3


"""
Flatten the directory structure of images and COLMAP model
- images will be all moved to the same directory - their relative path will be 
  preserved in their basename (replace / with _)
- COLMAP model will be adjusted to match the image paths
"""


import os
import argparse
import shutil
import pycolmap


parser = argparse.ArgumentParser(description="Flatten directory structure of images and COLMAP model")
parser.add_argument("--input_images", type=str,
                    help="Input images directory (the root of the directory structure - must match the COLMAP model)")
parser.add_argument("--output_images", type=str,
                    help="Output images directory")
parser.add_argument("--input_colmap", type=str,
                    help="Input COLMAP model directory")
parser.add_argument("--output_colmap", type=str,
                    help="Output COLMAP model directory")


def main(args):
    if args.input_colmap is not None and args.output_colmap is not None:
        print("- reading COLMAP model")
        model = pycolmap.Reconstruction(args.input_colmap)
        input_image_list = [img.name for img in model.images.values()]
        print(input_image_list)

        print("- adjusting COLMAP model")
        adjust_colmap_model(model)
        print("- writing COLMAP model")
        model.write_text(args.output_colmap)
    
    if args.input_images is not None and args.output_images is not None:
        print("- flattening images")
        flatten_images(args.input_images, args.output_images, input_image_list)


def adjust_colmap_model(model):
    for img in model.images.values():
        img.name = img.name.replace("/", "_")


def flatten_images(input_images, output_images, input_image_list=None):
    for dp, dn, filenames in os.walk(input_images):
        for f in filenames:
            src = os.path.join(dp, f)

            if input_image_list is not None:
                src_rel = os.path.relpath(src, input_images)
                if src_rel not in input_image_list:
                    continue

            dst = os.path.join(output_images, os.path.relpath(src, input_images).replace("/", "_"))
            shutil.copy(src, dst)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
