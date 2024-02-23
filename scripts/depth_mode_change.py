#!/usr/bin/env python


"""
Change the mode of depth file between "depth" as point Z ccordinate in 
the camera frame and "distance" as the point Euclidean distance from 
the camera center.
"""


import os
import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument(
    "--input_depth",
    type=str,
    required=True,
    help="path to input depth / distance file in a Numpy format",
)
parser.add_argument(
    "--output_depth",
    type=str,
    required=True,
    help="path to output depth / distance file in a Numpy format",
)
parser.add_argument(
    "--input_mode",
    type=str,
    choices=["depth", "distance"],
    default="distance",
    help="mode of the input data - depth / distance",
)
parser.add_argument(
    "--output_mode",
    type=str,
    choices=["depth", "distance"],
    default="depth",
    help="mode of the output data - depth / distance",
)
parser.add_argument(
    "--colmap_model",
    type=str,
    help="a COLMAP model to get the focal length of the images",
)
parser.add_argument(
    "--depth_postfix",
    type=str,
    default="",
    help="depth file postfix - used to determine image name in COLMAP model",
)
parser.add_argument(
    "--color_postfix",
    type=str,
    default="",
    help="color file postfix - used to determine image name in COLMAP model",
)
parser.add_argument(
    "--colmap_basenames",
    action="store_true",
    help="Change the image names in the COLMAP model just to their basenames",
)
parser.add_argument(
    "--focal_length", 
    type=float, 
    help="focal length"
)
parser.add_argument(
    "--dataparser_transforms",
    type=str,
    required=False,
    help="path to dataparser_transforms.json from nerfstudio - used for scaling of the depth map to the original training scale",
)
parser.add_argument(
    "--output_format",
    type=str,
    default="npz",
    choices=["npy", "npz"],
    help="output format - npy / npz",
)


def main(args):
    print("Transforming depth files:")
    scale = 1.0
    if args.dataparser_transforms is not None:
        assert os.path.isfile(
            args.dataparser_transforms
        ), "The dataparser_transforms.json file does not exist: {}".format(
            args.dataparser_transforms
        )

        import json

        with open(args.dataparser_transforms) as f:
            json_data_txt = f.read()

        json_data = json.loads(json_data_txt)
        scale = 1.0 / json_data["scale"]

        print("- loaded the scale from dataparser_transforms.json")

    if os.path.isdir(args.input_depth):
        assert os.path.isdir(
            args.output_depth
        ), "The output depth directory does not exist: {}".format(args.output_depth)
        input_depth_list = os.listdir(args.input_depth)
        input_depth_list = [
            os.path.join(args.input_depth, file)
            for file in input_depth_list
            if os.path.splitext(file)[1] in (".npy", ".npz", ".gz")
        ]
    elif os.path.isfile(args.input_depth):
        assert os.path.isfile(
            args.output_depth
        ), "The output depth file does not exist: {}".format(args.output_depth)
        input_depth_list = [args.input_depth]
    else:
        assert os.path.exists(
            args.input_depth
        ), "The input depth path does not exist: {}".format(args.input_depth)

    focal_lengths = {}

    if args.colmap_model is not None:
        assert os.path.isdir(
            args.colmap_model
        ), "The COLMAP model directory does not exist: {}".format(args.colmap_model)
        import pycolmap

        model = pycolmap.Reconstruction(args.colmap_model)

        if model.num_cameras() == 1:
            for _, cam in model.cameras.items():
                f = get_foclen(cam)
                break
            for file_path in input_depth_list:
                focal_lengths[file_path] = f
        else:
            model_name_map = {}

            for _, img in model.images.items():
                orig_name = img.name
                if args.colmap_basenames:
                    new_name = os.path.splitext(
                        os.path.basename(orig_name).replace("/", "_").replace(" ", "_")
                    )[0]
                else:
                    new_name = os.path.splitext(
                        orig_name.replace("/", "_").replace(" ", "_")
                    )[0]

                if args.color_postfix:
                    new_name = new_name[: -len(args.color_postfix)]
                model_name_map[new_name] = orig_name

            for file_path in input_depth_list:
                file_name, ext = os.path.splitext(
                    os.path.basename(file_path).replace("/", "_").replace(" ", "_")
                )
                if ext == ".gz":
                    file_name = os.path.splitext(file_name)[0]

                if args.depth_postfix:
                    file_name = file_name[: -len(args.depth_postfix)]

                if file_name not in model_name_map:
                    print("WARN: {} not found in the COLMAP model".format(file_name))
                    print("  - image names from COLMAP:")
                    print(model_name_map)
                    exit()

                orig_img_name = model_name_map[file_name]
                orig_img = model.find_image_with_name(orig_img_name)
                f = get_foclen(model.cameras[orig_img.camera_id])

                focal_lengths[file_path] = f

    else:
        assert (
            args.focal_length is not None
        ), "Please provide a focal length or COLMAP model"
        for file_path in input_depth_list:
            focal_lengths[file_path] = args.focal_length

    print("- scaling the depth by: {:.2f}".format(scale))
    print("- transforming {} --> {}".format(args.input_mode, args.output_mode))

    for input_depth in input_depth_list:
        f = focal_lengths[input_depth]

        depth_ext = os.path.splitext(input_depth)[1]
        if depth_ext == ".npy":
            depth_map = np.load(input_depth)
            depth_map = np.squeeze(depth_map)
        elif depth_ext == ".npz":
            # - if the input is in .npz format, take the "depth" or first key
            depth_map = np.load(input_depth)
            if "depth" in depth_map.keys():
                depth_map = depth_map["depth"]
            else:
                depth_map = depth_map[list(depth_map.keys())[0]]
            depth_map = np.squeeze(depth_map).astype(np.float32)
        elif depth_ext == ".gz":
            import gzip

            with gzip.open(input_depth, "rb") as gz:
                depth_map = np.load(gz)
            depth_map = np.squeeze(depth_map)

        if (depth_map.ndim > 2) and (depth_map.shape[2] > 1):
            depth_map = depth_map[:, :, 0]

        depth_map.squeeze()
        depth_map = np.ascontiguousarray(depth_map)

        X, Y = np.meshgrid(np.arange(depth_map.shape[1]), np.arange(depth_map.shape[0]))
        Y = Y.astype(np.float32) - (depth_map.shape[0] / 2.0)
        X = X.astype(np.float32) - (depth_map.shape[1] / 2.0)

        if args.input_mode == "depth":
            z = depth_map
        elif args.input_mode == "distance":
            z = f * depth_map / np.sqrt(Y**2 + X**2 + f**2)

        if args.output_mode == "depth":
            depth_map_out = scale * z
        elif args.output_mode == "distance":
            depth_map_out = scale * z * np.sqrt(Y**2 + X**2 + f**2) / f

        if os.path.isdir(args.input_depth):
            output_depth = os.path.join(
                args.output_depth, os.path.basename(input_depth)
            )
        else:
            output_depth = args.output_depth

        output_depth, ext = os.path.splitext(output_depth)
        if ext == ".gz":
            output_depth = os.path.splitext(output_depth)[0]

        output_depth = output_depth + "." + args.output_format

        if os.path.splitext(output_depth)[1] == ".npy":
            np.save(output_depth, depth_map_out)
        elif os.path.splitext(output_depth)[1] == ".npz":
            np.savez(output_depth, depth=depth_map_out)
        else:
            assert os.path.splitext(output_depth)[1] in (".npy", ".npz")


def get_foclen(cam):
    if len(cam.focal_length_idxs()) == 1:
        return cam.focal_length
    else:
        return cam.focal_length_x


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
