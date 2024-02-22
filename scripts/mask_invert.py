#!/usr/bin/env python

"""
Invert a binary mask (black/white) to the opposite format (white/black).
"""


import os
import argparse
from tqdm import tqdm
from PIL import Image, ImageOps
import numpy as np


parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument("input_mask_dir", type=str,
                    help="The input directory containing masks")
parser.add_argument("--output_mask_dir", type=str,
                    help="The output directory for the masks")


def main(args):
    assert os.path.isdir(args.input_mask_dir)

    list_dir = os.listdir(args.input_mask_dir)
    for filename in tqdm(list_dir):
        filepath = os.path.join(args.input_mask_dir, filename)
        
        if args.output_mask_dir is not None:
            assert os.path.isdir(args.output_mask_dir)
            out_filepath = os.path.join(args.output_mask_dir, filename)
        else:
            out_filepath = filepath

        try:
            ImageOps.invert(Image.open(filepath)).save(out_filepath)
        except IOError:
            print("- skipping non-image file: " + filename)    
            continue


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
