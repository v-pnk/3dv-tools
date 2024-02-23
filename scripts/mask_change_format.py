#!/usr/bin/env python3


"""
Change format of a binary segmentation mask

Supported formats:
                  black     channels    dtype   suffix      prefix
- Instant-NGP     pass      3           uint8   ---         dynamic_mask_
- NerfStudio      mask      1           uint8   ---         ---

"""


import os
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir", 
    type=str, 
    help="The directory with the input masks"
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    help="The directory where the output masks will be stored"
)
parser.add_argument(
    "--input_format", 
    type=str, 
    choices=["ingp", "ns"], 
    help="Input mask format"
)
parser.add_argument(
    "--output_format", 
    type=str, 
    choices=["ingp", "ns"], 
    help="Output mask format"
)
parser.add_argument(
    "--input_mask_suffix",
    type=str,
    default="",
    help="Convert only files with this suffix - the suffix will be removed in the output files",
)
parser.add_argument(
    "--input_mask_prefix",
    type=str,
    default="",
    help="Convert only files with this prefix - the prefix will be removed in the output files",
)
parser.add_argument(
    "--output_mask_suffix", 
    type=str, 
    default="", 
    help="Add suffix to the output files"
)
parser.add_argument(
    "--output_mask_prefix",
    type=str,
    default="",
    help="Add prefix to the output files",
)


def main(args: argparse.Namespace):
    assert os.path.isdir(args.input_dir)
    assert os.path.isdir(args.output_dir)

    file_list = os.listdir(args.input_dir)
    file_list = [file for file in file_list if os.path.splitext(file)[1] == ".png"]
    file_list = [
        file
        for file in file_list
        if os.path.splitext(file)[0].endswith(args.input_mask_suffix)
    ]
    file_list = [file for file in file_list if file.startswith(args.input_mask_prefix)]
    file_list.sort()

    for file in tqdm(file_list):
        in_file_path = os.path.join(args.input_dir, file)
        file_name = os.path.splitext(os.path.basename(in_file_path))[0]
        file_name = file_name[
            len(args.input_mask_prefix) : -len(args.input_mask_suffix)
        ]
        file_name = args.output_mask_prefix + file_name + args.output_mask_suffix
        out_file_path = os.path.join(args.output_dir, file_name + ".png")

        img_in = Image.open(in_file_path)
        img_np = np.asarray(img_in)

        # Transform the input mask to a base format (equal to NS format)
        # - black is mask, single channel, uint8
        if args.input_format == "ingp":
            img_np = img_np[:, :, 0].squeeze()
            img_np = 255 - img_np

        # Transform the base format to the output mask
        if args.output_format == "ingp":
            img_np = 255 - img_np
            img_np = np.tile(img_np[:, :, None], (1, 1, 3))

        img_out = Image.fromarray(img_np)
        img_out.save(out_file_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
