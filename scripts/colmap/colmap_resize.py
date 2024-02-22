#!/usr/bin/env python3


"""
Resizes images and adjusts cameras and feature points in the given COLMAP model.
"""


import os
import argparse
import shutil
from PIL import Image
import pycolmap


parser = argparse.ArgumentParser(description="Resize images and COLMAP model")
parser.add_argument("--input_images", type=str, help="Input images directory")
parser.add_argument("--output_images", type=str, help="Output images directory")
parser.add_argument("--input_colmap", type=str, help="Input COLMAP model directory")
parser.add_argument("--output_colmap", type=str, help="Output COLMAP model directory")
parser.add_argument(
    "--max_size", type=int, help="The max size (longer side) of output image"
)
parser.add_argument(
    "--size_ratio",
    type=float,
    help="Resize by a size ratio (e.g. 0.5 means resize to half the size)",
)
parser.add_argument(
    "--resize_by_images",
    action="store_true",
    help="Resize the COLMAP model based on sizes of the image in the input_images directory",
)
parser.add_argument(
    "--resize_by_colmap",
    action="store_true",
    help="Resize the images based on sizes of the COLMAP modelin the input_colmap directory",
)
parser.add_argument(
    "--image_interp", type=str, choices=["bicubic", "nearest"], 
    default="bicubic", 
    help="Resize the images based on sizes of the COLMAP modelin the input_colmap directory",
)


def main(args):
    num_size_specs = 0
    if args.max_size is not None:
        num_size_specs += 1
    if args.size_ratio is not None:
        num_size_specs += 1
    if args.resize_by_images:
        assert (
            args.input_images is not None
        ), "Please specify input_images directory to use resize_by_images argument"
        assert (args.input_colmap is not None) and (
            args.output_colmap is not None
        ), "Please specify input_colmap and output_colmap directories to use resize_by_images argument"
        num_size_specs += 1
    if args.resize_by_colmap:
        assert (
            args.input_colmap is not None
        ), "Please specify input_colmap directory to use resize_by_colmap argument"
        assert (args.input_images is not None) and (
            args.output_images is not None
        ), "Please specify input_images and output_images directories to use resize_by_colmap argument"
        num_size_specs += 1

    assert (
        num_size_specs == 1
    ), "Please use exactly one of max_size, size_ratio, resize_by_images, or resize_by_colmap arguments"

    if args.input_colmap is not None:
        print("- reading the input COLMAP model")
        model = pycolmap.Reconstruction(args.input_colmap)

    if args.resize_by_images:
        img_sizes = get_sizes_from_images(args.input_images)
    if args.resize_by_colmap:
        img_sizes = get_sizes_from_colmap(model)

    if (
        (args.input_images is not None)
        and (args.output_images is not None)
        and not args.resize_by_images
    ):
        print("- resizing the images")
        file_list = [
            os.path.relpath(os.path.join(dp, f), args.input_images)
            for dp, dn, filenames in os.walk(args.input_images)
            for f in filenames
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))
        ]

        for img_in_file in file_list:
            img_in_path = os.path.join(args.input_images, img_in_file)
            img_out_path = os.path.join(args.output_images, img_in_file)
            img_in = Image.open(img_in_path)

            if args.max_size is not None:
                img_in_max_size = max(img_in.size)
                if img_in_max_size < args.max_size:
                    shutil.copy(img_in_path, img_out_path)
                    continue

                resize_ratio = 1.0 * args.max_size / img_in_max_size
                new_width = int(round(resize_ratio * img_in.size[0]))
                new_height = int(round(resize_ratio * img_in.size[1]))
            elif args.size_ratio is not None:
                resize_ratio = args.size_ratio
                new_width = int(round(resize_ratio * img_in.size[0]))
                new_height = int(round(resize_ratio * img_in.size[1]))
            elif args.resize_by_colmap:
                new_width = img_sizes[img_in_file][0]
                new_height = img_sizes[img_in_file][1]

            if args.image_interp == "bicubic":
                img_out = img_in.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
            elif args.image_interp == "nearest":
                img_out = img_in.resize((new_width, new_height), resample=Image.Resampling.NEAREST)

            img_out_dir = os.path.dirname(img_out_path)
            if not (os.path.exists(img_out_dir)):
                os.makedirs(img_out_dir)

            img_out.save(img_out_path)

    if (args.input_colmap is not None) and (args.output_colmap is not None):
        print("- resizing the COLMAP model")
        cam_sizes = {}

        # rescale the 2D points
        for img in model.images.values():
            img_cam = model.cameras[img.camera_id]
            cam_w = img_cam.width
            cam_h = img_cam.height

            if args.max_size is not None:
                cam_max_size = max(cam_w, cam_h)
                if cam_max_size < args.max_size:
                    continue

                resize_ratio = args.max_size / cam_max_size
            elif args.size_ratio is not None:
                resize_ratio = args.size_ratio
            elif args.resize_by_images:
                resize_ratio = 1.0 * img_sizes[img.name][0] / cam_w
                cam_sizes[img.camera_id] = (
                    img_sizes[img.name][0],
                    img_sizes[img.name][1],
                )

            for pnt2D in img.points2D:
                pnt2D.x = resize_ratio * pnt2D.x
                pnt2D.y = resize_ratio * pnt2D.y

        # rescale the cameras
        for cam in model.cameras.values():
            cam_w = cam.width
            cam_h = cam.height

            if args.max_size is not None:
                cam_max_size = max(cam_w, cam_h)
                if cam_max_size < args.max_size:
                    continue

                resize_ratio = args.max_size / cam_max_size
                cam_w_new = int(round(resize_ratio * cam_w))
                cam_h_new = int(round(resize_ratio * cam_h))
            elif args.size_ratio is not None:
                resize_ratio = args.size_ratio
                cam_w_new = int(round(resize_ratio * cam_w))
                cam_h_new = int(round(resize_ratio * cam_h))
            elif args.resize_by_images:
                if cam.camera_id not in cam_sizes:
                    continue
                cam_w_new = cam_sizes[cam.camera_id][0]
                cam_h_new = cam_sizes[cam.camera_id][1]

            cam.rescale(cam_w_new, cam_h_new)

        print("- writing COLMAP model")
        model.write_text(args.output_colmap)


def get_sizes_from_images(image_dir):
    img_sizes = {}
    for img_in_file in os.listdir(image_dir):
        img_in_path = os.path.join(image_dir, img_in_file)
        img_in = Image.open(img_in_path)
        img_sizes[img_in_file] = img_in.size
    return img_sizes


def get_sizes_from_colmap(model):
    img_sizes = {}
    for img in model.images.values():
        img_cam = model.cameras[img.camera_id]
        cam_w = img_cam.width
        cam_h = img_cam.height
        img_sizes[img.name] = (cam_w, cam_h)
    return img_sizes


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
