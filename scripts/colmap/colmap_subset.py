#!/usr/bin/env python3


"""
Adjusts COLMAP model so it contains only the images in the specified directory
or list file.
"""


import os
import copy
import argparse
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_colmap", 
    type=str, 
    help="Input COLMAP model directory"
)
parser.add_argument(
    "output_colmap", 
    type=str, 
    help="Output COLMAP model directory"
)
parser.add_argument(
    "--new_img_dir",
    type=str,
    help="Image directory with the subset of images"
)
parser.add_argument(
    "--new_img_list",
    type=str,
    help="Image list file (one image relative path per line) with the subset of images",
)
parser.add_argument(
    "--every_nth",
    type=int,
    help="Use every nth image (in alphabetical order) from the input_colmap model",
)
parser.add_argument(
    "--old_img_postfix",
    type=str,
    help="Filename postfix of the images in the input_colmap model",
)
parser.add_argument(
    "--ignore_old_img_ext", 
    action="store_true", 
    help="Ignore extensions of old images"
)
parser.add_argument(
    "--new_img_postfix",
    type=str,
    help="Filename postfix of the images in the input_img_dir",
)
parser.add_argument(
    "--underscores",
    action="store_true",
    help="Match image paths with slash characters replaced by underscores",
)


def main(args):
    assert os.path.isdir(
        args.input_colmap
    ), "The input COLMAP model does not exist: {}".format(args.input_colmap)
    assert os.path.isdir(
        args.output_colmap
    ), "The output COLMAP model does not exist: {}".format(args.output_colmap)

    print("- reading COLMAP model")
    model_in = pycolmap.Reconstruction(args.input_colmap)

    print("- getting the list of images")
    img_mapping = {}

    if args.ignore_old_img_ext:
        args.old_img_postfix = ""

    if args.new_img_dir is not None:
        assert os.path.isdir(
            args.new_img_dir
        ), "The new image directory does not exist: {}".format(args.new_img_dir)

        new_images = [
            os.path.relpath(os.path.join(dp, f), args.new_img_dir)
            for dp, _, filenames in os.walk(args.new_img_dir)
            for f in filenames
        ]
    elif args.new_img_list is not None:
        assert os.path.isfile(
            args.new_img_list
        ), "The new image list file does not exist: {}".format(args.new_img_list)

        new_images = read_img_list(args.new_img_list)
    elif args.every_nth is not None:
        old_images = [img.name for img in model_in.images.values()]
        new_images = sorted(old_images)[:: args.every_nth]
    else:
        raise ValueError("Either new_img_dir, new_img_list or every_nth value must be specified")

    if (args.old_img_postfix is not None) and (args.new_img_postfix is not None):
        for new_img in new_images:
            old_img = new_img[: -len(args.new_img_postfix)] + args.old_img_postfix
            img_mapping[old_img] = new_img
    else:
        for img in new_images:
            img_mapping[img] = img

    print("- filtering COLMAP model images")
    model_out = copy.deepcopy(model_in)

    found_img_num = 0

    for img in model_in.images.values():
        img_name = img.name

        if args.underscores:
            img_name = img_name.replace("/", "_")

        if args.ignore_old_img_ext:
            img_name = os.path.splitext(img_name)[0]

        if img_name not in img_mapping:
            model_out.deregister_image(img.image_id)
        else:
            found_img_num += 1

    print("  - found {} images".format(found_img_num))

    print("- writing new COLMAP model")
    model_out.write(args.output_colmap)


def read_img_list(img_list_path):
    with open(img_list_path, "rt") as f:
        line_list = f.readlines()
    img_list = [line.strip() for line in line_list]
    return img_list


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
