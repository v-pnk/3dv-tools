#!/usr/bin/env python3


"""
Blend a binary with a corresponding image, with an option to choose color and
weight of the mask.
"""


import os
import argparse
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument(
    "input_masks", 
    type=str, 
    help="The directory with the input masks",
)
parser.add_argument(
    "input_images", 
    type=str, 
    help="The directory with the input images",
)
parser.add_argument(
    "output_images",
    type=str,
    help="The directory where the output images will be stored",
)
parser.add_argument(
    "--mode",
    type=str,
    choices=["black_to_0", "white_to_0"],
    default="black_to_0",
    help="Mode of mask conversion",
)
parser.add_argument(
    "--mask_color",
    type=int,
    nargs=3,
    default=[255, 0, 0],
    help="Mask color as triple of uint8 values [0-255]",
)
parser.add_argument(
    "--mask_weight",
    type=float,
    default=0.5,
    help="Weight of the mask color",
)
parser.add_argument(
    "--input_mask_suffix", 
    type=str, default="", 
    help="Input mask file name suffix"
)
parser.add_argument(
    "--input_image_suffix", 
    type=str, 
    default="", 
    help="Input image file name suffix"
)
parser.add_argument(
    "--output_image_suffix", 
    type=str, 
    default="", 
    help="Output image file name suffix"
)


def main(args: argparse.Namespace):
    mask_path_list = []
    img_path_list = []
    output_path_list = []

    if os.path.isdir(args.input_masks):
        assert os.path.isdir(args.input_images)
        assert os.path.isdir(args.output_images)
        mask_list = os.listdir(args.input_masks)
        img_list = os.listdir(args.input_images)
        mask_list.sort()
        img_list.sort()
        img_base_list = [os.path.splitext(img)[0] for img in img_list]

        if args.input_mask_suffix != "":
            args.input_mask_suffix = os.path.splitext(args.input_mask_suffix)[0]
            if args.input_mask_suffix != "":
                mask_list = [
                    mask
                    for mask in mask_list
                    if os.path.splitext(mask)[0].endswith(args.input_mask_suffix)
                ]

        if args.input_image_suffix != "":
            args.input_image_suffix = os.path.splitext(args.input_image_suffix)[0]
            img_base_list = [
                img_base[-len(args.input_image_suffix) :]
                for img_base in img_base_list
                if img_base.endswith(args.input_image_suffix)
            ]

        if args.output_image_suffix != "":
            args.output_image_suffix = os.path.splitext(args.output_image_suffix)[0]

        for mask in mask_list:
            if args.input_mask_suffix:
                name = os.path.splitext(mask)[0][: -len(args.input_mask_suffix)]
            else:
                name = os.path.splitext(mask)[0]

            if name not in img_base_list:
                print("WARN: Mask {} does not have a corresponding image".format(mask))
                continue

            img_idx = img_base_list.index(name)
            mask_path = os.path.join(args.input_masks, mask)
            img_path = os.path.join(args.input_images, img_list[img_idx])
            out_path = os.path.join(
                args.output_images,
                img_base_list[img_idx] + args.output_image_suffix + ".jpg",
            )

            mask_path_list.append(mask_path)
            img_path_list.append(img_path)
            output_path_list.append(out_path)

    elif os.path.isfile(args.input_masks):
        assert os.path.isfile(args.input_images)
        mask_path_list = [args.input_masks]
        img_path_list = [args.input_images]
        output_path_list = [args.output_images]

    for mask_path, img_path, out_path in tqdm(
        zip(mask_path_list, img_path_list, output_path_list)
    ):
        image_in = Image.open(img_path)
        image_in = image_in.convert("RGB")
        mask_in = Image.open(mask_path)
        mask_in = mask_in.convert("L")

        if args.mode == "white_to_0":
            mask_in = ImageOps.invert(mask_in)

        mask_in_np = np.array(mask_in)
        mask_in_color = np.tile(
            np.array(args.mask_color)[None, None, :],
            (mask_in_np.shape[0], mask_in_np.shape[1], 1),
        )
        mask_in_np = np.tile(mask_in_np[:, :, None], (1, 1, 3))

        image_in_np = np.array(image_in).astype(np.float32)
        image_in_np_masked = np.copy(image_in_np)
        image_in_np_masked[mask_in_np == 0] = mask_in_color[mask_in_np == 0]
        image_in_np = (
            args.mask_weight * image_in_np_masked + (1 - args.mask_weight) * image_in_np
        )
        image_in_np = image_in_np.astype(np.uint8)

        image_in = Image.fromarray(image_in_np)
        image_in.save(out_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
