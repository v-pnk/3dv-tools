#!/usr/bin/env python3


"""
Visualize depth map as an image where the depths are encoded by a colormap or 
as a point cloud. The depth map can be in .npy, .npz, .png or .h5 format. The
coloring of the point cloud can be based on the depth map or on the input RGB
image. The focal length is required for the point cloud visualization.

"""


import os
import argparse

import open3d as o3d
import numpy as np
import matplotlib as mpl
from PIL import Image


parser = argparse.ArgumentParser(description="Argument parser")
parser.add_argument(
    "input_depth",
    type=str,
    help="Path to the input depth map in .npy, .npz, .png or .h5 format",
)
parser.add_argument(
    "--input_rgb",
    type=str,
    required=False,
    help="Path to input RGB image - can be used to color the point cloud",
)
parser.add_argument(
    "--focal_length",
    type=float,
    help="Camera focal length used for the back-projection to point cloud",
)

parser.add_argument(
    "--visualize_depth", 
    action="store_true", 
    help="Visualize the depth map as image"
)
parser.add_argument(
    "--visualize_pc",
    action="store_true",
    help="Visualize point cloud created from the depth map",
)

parser.add_argument(
    "--depth_mode",
    type=str,
    choices=["depth", "distance"],
    default="depth",
    help='Switch between the input depth type - "depth" is Z distance from camera center, "distance" is Euclidean distance from the camera center',
)
parser.add_argument(
    "--inverse_depth",
    action="store_true",
    help="Inverse inverse the input depth values as 1/d",
)
parser.add_argument(
    "--reverse_color", 
    action="store_true", 
    help="Reverse the used colormap"
)
parser.add_argument(
    "--depth_multiplier", 
    type=float, 
    default=1.0, 
    help="Depth multiplier"
)
parser.add_argument(
    "--nan2zero", 
    action="store_true", 
    help="Turn all NaN values to 0"
)

parser.add_argument(
    "--palette_min", 
    type=float, 
    help="Minimal values visible in color palette"
)
parser.add_argument(
    "--palette_max", 
    type=float, 
    help="Maximal values visible in color palette"
)
parser.add_argument(
    "--palette_hide_out",
    action="store_true",
    help="Hide elements outside of set palette limits",
)
parser.add_argument(
    "--background_color",
    type=float,
    default=[255 / 255, 255 / 255, 255 / 255],
    nargs="+",
    help="Background color of the visualization - default: %(default)s",
)

parser.add_argument(
    "--output_image", 
    type=str, 
    help="Path to the output depth image"
)
parser.add_argument(
    "--output_pc", 
    type=str, 
    help="Path to the output point cloud"
)


def main(args):
    # - read the depth map
    depth_ext = os.path.splitext(args.input_depth)[1]
    if depth_ext == ".png":
        depth_img = o3d.io.read_image(args.input_depth)
        depth_img = np.asarray(depth_img)
    elif depth_ext == ".npy":
        depth_img = np.load(args.input_depth)
        depth_img = np.squeeze(depth_img)
    elif depth_ext == ".npz":
        # - if the input is in .npz format, take the first key
        depth_img = np.load(args.input_depth)
        depth_img = depth_img[list(depth_img.keys())[0]]
        depth_img = np.squeeze(depth_img).astype(np.float32)
    elif depth_ext == ".h5":
        import h5py

        f = h5py.File(args.input_depth)
        depth_img = np.asarray(f["depth"]).astype(np.float32)
    else:
        assert False, "Unknown depth_img extension: {}".format(depth_ext)

    if (depth_img.ndim > 2) and (depth_img.shape[2] > 1):
        depth_img = depth_img[:, :, 0]

    depth_img = np.ascontiguousarray(depth_img)

    if args.nan2zero:
        depth_img[np.isnan(depth_img)] = 0

    if args.inverse_depth:
        depth_img[abs(depth_img) > 1e-6] = 1.0 / depth_img[abs(depth_img) > 1e-6]

    depth_img = args.depth_multiplier * depth_img

    depth_img = np.expand_dims(depth_img, axis=2)
    depth_img = depth_img.astype(np.float32)
    depth_raw = o3d.geometry.Image(depth_img)
    depth_width = depth_img.shape[0]
    depth_height = depth_img.shape[1]
    depth_min = np.amin(depth_img)
    depth_max = np.amax(depth_img)

    print("- min depth in file = {:.3f}".format(depth_min))
    print("- max depth in file = {:.3f}".format(depth_max))

    # - color the point cloud based on the file from args.input_rgb in .npy format
    if args.input_rgb is not None and os.path.splitext(args.input_rgb)[1] == ".npy":
        # - read the .npy data and apply colormap on them
        color_data = np.load(args.input_rgb)

        min_value = np.amin(color_data)
        max_value = np.amax(color_data)

        print("- min rgb value = {:.3f}".format(min_value))
        print("- max rgb value = {:.3f}".format(max_value))

        if args.palette_min is not None:
            min_value = args.palette_min
        if args.palette_max is not None:
            max_value = args.palette_max

        values_norm = (color_data - min_value) / (max_value - min_value)
        values_norm = np.clip(values_norm, 0, 1)

        if args.reverse_color:
            values_norm = 1.0 - values_norm

        depth_img_clr = np.zeros((depth_width, depth_height, 3))

        cmap = mpl.colormaps["turbo"]
        depth_img_clr = cmap(values_norm.astype(np.float32)).squeeze()[:, :, 0:3]
        depth_img_clr = (255 * depth_img_clr).astype(np.uint8)

        print("- coloring the point cloud based on the data in the color .npy file")
        visualize_colormap(args.reverse_color, min_value, max_value)

    elif args.input_rgb is not None and os.path.splitext(args.input_rgb)[1] != ".npy":
        # - read the RGB image and adjust it to the size of the depth map
        depth_img_clr = Image.open(args.input_rgb)
        depth_img_clr = depth_img_clr.convert("RGB")
        depth_img_clr = depth_img_clr.resize((depth_height, depth_width))
        depth_img_clr = np.array(depth_img_clr)

        print("- coloring the point cloud based on the color image")
    else:
        depth_raw_imglike = depth_img

        # ignore depth = 0 (usually invalid)
        depth_invalid = depth_img == 0
        min_depth = np.amin(depth_raw_imglike[np.logical_not(depth_invalid)])
        max_depth = np.amax(depth_raw_imglike)

        if args.palette_min is not None:
            min_depth = args.palette_min
        if args.palette_max is not None:
            max_depth = args.palette_max

        depth_raw_imglike = (depth_raw_imglike - min_depth) / (max_depth - min_depth)
        depth_raw_imglike = np.clip(depth_raw_imglike, 0, 1)

        if args.reverse_color:
            depth_raw_imglike = 1.0 - depth_raw_imglike

        # - if not present, set color by depth
        depth_img_clr = np.zeros((depth_width, depth_height, 3))

        cmap = mpl.colormaps["turbo"]
        depth_img_clr = cmap(depth_raw_imglike.astype(np.float32)).squeeze()[:, :, 0:3]
        depth_img_clr[np.tile(depth_invalid, (1, 1, 3))] = 0
        depth_img_clr = (255 * depth_img_clr).astype(np.uint8)

        print("- coloring the point cloud based on the depth")
        print("  - the depth values equal to 0 are filtered out")
        visualize_colormap(args.reverse_color, min_depth, max_depth)

    color_raw = o3d.geometry.Image(np.array(depth_img_clr).astype(np.uint8))

    if args.visualize_pc and args.focal_length is None:
        assert False, "Focal length is required for point cloud visualization"

    if args.visualize_depth or (args.output_image is not None):
        if args.output_image is not None:
            Image.fromarray(depth_img_clr).save(args.output_image)

        if args.visualize_depth:
            Image.fromarray(depth_img_clr).show()

    if args.visualize_pc or (args.output_pc is not None):
        depth_map = depth_img.squeeze()
        Y, X = np.where(depth_map > 0.0)
        num_valid = np.count_nonzero(depth_map)
        xyz = np.ones((num_valid, 3))
        xyz[:, 0] = X.astype(np.float32) - depth_map.shape[1] / 2.0
        xyz[:, 1] = Y.astype(np.float32) - depth_map.shape[0] / 2.0
        xyz[:, 2] = args.focal_length

        if args.depth_mode == "depth":
            xyz = xyz * depth_map[Y, X][:, None] / args.focal_length
        elif args.depth_mode == "distance":
            xyz = (
                xyz
                * depth_map[Y, X][:, None]
                / np.tile(np.sqrt(np.sum(xyz**2, axis=1, keepdims=True)), (1, 3))
            )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        depth_raw_np_flat = np.asarray(depth_raw).flatten()
        color_raw_np_flat = np.reshape(
            np.swapaxes(np.asarray(color_raw) / 255.0, 0, 2), (3, -1), order="F"
        )
        valid_idx = np.logical_and(
            depth_raw_np_flat > 0.0, np.logical_not(np.isinf(depth_raw_np_flat))
        )
        color_raw_np_flat = color_raw_np_flat[:, valid_idx]
        pcd.colors = o3d.utility.Vector3dVector(color_raw_np_flat.T)

        # - estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.03, max_nn=20)
        )
        # - normalize normals to unit vectors
        pcd.normalize_normals()

        if args.output_pc is not None:
            o3d.io.write_point_cloud(args.output_pc, pcd)

        if args.visualize_pc:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(pcd)
            vis.get_render_option().background_color = np.array(args.background_color)
            vis.run()


def visualize_colormap(reverse_color, min_value, max_value):
    colormap_text = ""
    color_strings = []
    colormap_vals = np.linspace(0, 1, 32)
    cmap = mpl.colormaps["turbo"]
    if reverse_color:
        colormap_vals = np.flip(colormap_vals)
    for val_i in colormap_vals:
        col_i = (255.0 * np.array(cmap(val_i)[0:3])).astype(np.uint8)
        color_strings.append(
            "\x1b[38;2;{};{};{}m".format(col_i[0], col_i[1], col_i[2])
            + "\u2588"
            + "\033[0m"
        )
    colormap_text = "- colormap: min " + "".join(color_strings) + " max"

    min_max_text = " " * 12 + "{:<20.3f}{:20.3f}".format(min_value, max_value)

    print(colormap_text)
    print(min_max_text)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
