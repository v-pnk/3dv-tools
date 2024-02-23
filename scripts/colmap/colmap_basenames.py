#!/usr/bin/env python3


"""
Rename images in the COLMAP model to contain only basenames (without directories)

"""


import os
import argparse
import pycolmap


parser = argparse.ArgumentParser(
    description="Rename images in the COLMAP model to contain only basenames"
)
parser.add_argument(
    "--input_colmap", 
    type=str, 
    help="Input COLMAP model directory"
)
parser.add_argument(
    "--output_colmap", 
    type=str, 
    help="Output COLMAP model directory"
)
parser.add_argument(
    "--output_mode",
    type=str,
    choices=["TXT", "BIN"],
    default="TXT",
    help="Output COLMAP model format - TXT / BIN",
)


def main(args):
    assert os.path.isdir(args.input_colmap)
    assert os.path.isdir(args.output_colmap)

    model = pycolmap.Reconstruction(args.input_colmap)

    for image_id in model.images.keys():
        image = model.images[image_id]
        image.name = os.path.basename(image.name)
        model.images[image_id] = image

    if args.output_mode == "TXT":
        model.write_text(args.output_colmap)
    elif args.output_mode == "BIN":
        model.write_binary(args.output_colmap)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
