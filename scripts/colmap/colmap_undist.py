#!/usr/bin/env python3

"""
Undistort images and COLMAP model
"""

import os
import shutil
import argparse
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument("image_path", type=str, help="Image directory")
parser.add_argument("model_path", type=str, help="COLMAP model directory")
parser.add_argument("workdir", type=str, help="Working directory")
parser.add_argument("--undist_image_path", type=str, help="Undistorted image directory")
parser.add_argument("--undist_model_path", type=str, help="Undistorted COLMAP model directory")
parser.add_argument("--rm_workdir", action="store_true", help="Remove working directory after completion")
parser.add_argument("--output_type", type=str, required=False, 
                    choices=["BIN", "TXT"], default="BIN", 
                    help="Type of the output model")


def main(args):
    if not os.path.isdir(args.workdir):
        os.makedirs(args.workdir)
    
    print("- undistorting images and COLMAP model")
    pycolmap.undistort_images(args.workdir, args.model_path, args.image_path)

    
    if args.undist_image_path is not None:
        if args.rm_workdir:
            print("- moving undistorted images to {}".format(args.undist_image_path))
            move_dir(os.path.join(args.workdir, "images"), args.undist_image_path)
            # shutil.move(os.path.join(args.workdir, "images"), args.undist_image_path) # moves image dir inside the target if target exists
        else:
            print("- copying undistorted images to {}".format(args.undist_image_path))
            shutil.copytree(os.path.join(args.workdir, "images"), args.undist_image_path , dirs_exist_ok=True)
    
    if args.undist_model_path is not None:
        print("- copying undistorted COLMAP model to {}".format(args.undist_model_path))
        undist_model = pycolmap.Reconstruction(os.path.join(args.workdir, "sparse"))
        if args.output_type == "TXT":
            undist_model.write_text(args.undist_model_path)
        else:
            undist_model.write_binary(args.undist_model_path)

    if args.rm_workdir:
        print("- removing working directory {}".format(args.workdir))
        shutil.rmtree(args.workdir)


def move_dir(src, dst):
    for file in os.listdir(src):
        shutil.move(os.path.join(src, file), dst)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)