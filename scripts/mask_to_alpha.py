#!/usr/bin/env python3


"""
Move binary masks to alpha channel of PNG image
"""


import os
import argparse
from PIL import Image, ImageOps
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("input_masks", type=str, help="The directory with the input masks")
parser.add_argument("input_images", type=str, help="The directory with the input images")
parser.add_argument("output_images", type=str, help="The directory where the output images will be stored")
parser.add_argument("--mode", type=str, choices=["black_to_0", "white_to_0"], default="black_to_0", help="Mode of mask conversion")
parser.add_argument("--input_mask_suffix", type=str, default="", help="Input mask suffix")
parser.add_argument("--input_image_suffix", type=str, default="", help="Input image suffix")
parser.add_argument("--output_image_suffix", type=str, default="", help="Output image suffix")


def main(args: argparse.Namespace):
    assert os.path.isdir(args.input_masks)
    assert os.path.isdir(args.input_images)
    assert os.path.isdir(args.output_images)
    
    mask_list = os.listdir(args.input_masks)
    mask_list.sort()
    img_list = os.listdir(args.input_images)
    img_list.sort()
    img_base_list = [os.path.splitext(img)[0] for img in img_list]

    if args.input_mask_suffix != "":
        args.input_mask_suffix = os.path.splitext(args.input_mask_suffix)[0]
        if args.input_mask_suffix != "":
            mask_list = [mask for mask in mask_list if os.path.splitext(mask)[0].endswith(args.input_mask_suffix)]
    
    if args.input_image_suffix != "":
        args.input_image_suffix = os.path.splitext(args.input_image_suffix)[0]
        img_base_list = [img_base[-len(args.input_image_suffix):] for img_base in img_base_list if img_base.endswith(args.input_image_suffix)]

    if args.output_image_suffix != "":
        args.output_image_suffix = os.path.splitext(args.output_image_suffix)[0]
    
    for mask in tqdm(mask_list):
        if args.input_mask_suffix:
            name = os.path.splitext(mask)[0][:-len(args.input_mask_suffix)]
        else:
            name = os.path.splitext(mask)[0]
            
        if name not in img_base_list:
            print("WARN: Mask {} does not have a corresponding image".format(mask))
            continue
        
        img_idx = img_base_list.index(name)
        mask_path = os.path.join(args.input_masks, mask)
        img_path = os.path.join(args.input_images, img_list[img_idx])
        out_path = os.path.join(args.output_images, img_base_list[img_idx] + args.output_image_suffix + ".png")

        image_in = Image.open(img_path)
        image_in = image_in.convert("RGB")
        mask_in = Image.open(mask_path)
        mask_in = mask_in.convert("L")

        if args.mode == "white_to_0":
            mask_in = ImageOps.invert(mask_in)

        image_in.putalpha(mask_in)

        image_in.save(out_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)