"""
Apply a Lookup Table stored in .cube format or as a Hald image to a given input 
image or a directory with images.

Using pillow-lut package: https://github.com/homm/pillow-lut-tools
"""


import os
import argparse
from tqdm import tqdm

from PIL import Image
from pillow_lut import load_cube_file, load_hald_image


IMG_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff", ".bmp", ".gif"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_in",
                        type=str, 
                        help="Path to the input image or a directory with images")
    parser.add_argument("lut_file", 
                        type=str, 
                        help="Path to the LUT file")
    parser.add_argument("image_out", 
                        type=str, 
                        help="Path to the output image or a directory")
    args = parser.parse_args()

    assert os.path.exists(args.lut_file), f"LUT file {args.lut_file} does not exist"

    lut = load_lut(args.lut_file)

    if os.path.isdir(args.image_in):
        imgs_in_list = [os.path.join(args.image_in, f) for f in os.listdir(args.image_in) if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS]
        imgs_in_list.sort()
        assert os.path.isdir(args.image_out), f"If the input is a directory, the output must also be a directory"
        imgs_out_list = [os.path.join(args.image_out, os.path.basename(f)) for f in imgs_in_list]
    else:
        imgs_in_list = [args.image_in]
        if os.path.isdir(args.image_out):
            imgs_out_list = [os.path.join(args.image_out, os.path.basename(imgs_in_list[0]))]
        else:
            imgs_out_list = [args.image_out]

    for img_in, img_out in tqdm(zip(imgs_in_list, imgs_out_list), total=len(imgs_in_list)):
        img = Image.open(img_in)
        img = img.filter(lut)
        img.save(img_out)


def load_lut(lut_file):
    if lut_file.endswith(".cube"):
        # - pillow-lut 1.0.1 does not support .cube files containing tabs
        #   --> replace with spaces
        tmp_lut_file = f"{lut_file}.tmp"
        with open(lut_file, "r") as f:
            lut_data = f.read()
        lut_data = lut_data.replace("\t", " ")
        with open(tmp_lut_file, "w") as f:
            f.write(lut_data)
        lut = load_cube_file(tmp_lut_file)
        os.remove(tmp_lut_file)
        return lut
    elif os.path.splitext(lut_file)[1].lower() in IMG_EXTENSIONS:
        return load_hald_image(lut_file)
    else:
        raise ValueError(f"Unsupported LUT file format: {lut_file}")


if __name__ == "__main__":
    main()