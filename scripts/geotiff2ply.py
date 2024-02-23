#!/usr/bin/env python3


"""
Convert a surface height map in TIFF format to a PLY file.
"""


import argparse
import numpy as np
from skimage import io
import open3d as o3d


parser = argparse.ArgumentParser(description="")
parser.add_argument("input_tiff", type=str, help="Path to the input TIFF file")
parser.add_argument("output_ply", type=str, help="Path to the output PLY file")
parser.add_argument("--output_png", type=str, help="Path to the output PNG image")
parser.add_argument("--crop_left", type=int, help="Crop left side at this column")
parser.add_argument("--crop_right", type=int, help="Crop right side at this column")
parser.add_argument("--crop_bottom", type=int, help="Crop bottom side at this row")
parser.add_argument("--crop_top", type=int, help="Crop top side at this row")


def main(args):
    data = io.imread(args.input_tiff)

    height = data.shape[0]
    width = data.shape[1]
    print("TIFF data shape: {} x {}".format(width, height))

    left = 0
    right = width
    bottom = 0
    top = height

    if args.crop_left is not None:
        left = args.crop_left
    if args.crop_right is not None:
        right = args.crop_right
    if args.crop_bottom is not None:
        bottom = args.crop_bottom
    if args.crop_top is not None:
        top = args.crop_top

    assert left >= 0
    assert right <= width
    assert bottom >= 0
    assert top <= height
    assert left < right
    assert bottom < top

    crop_width = right - left
    crop_height = top - bottom

    data_crop = data[left:right, bottom:top]
    xpc, ypc = np.meshgrid(np.arange(left, right, 1), np.arange(bottom, top, 1))

    xpc = xpc.T
    ypc = ypc.T

    colors = (data_crop.T.flatten() - data_crop.min()) / (
        data_crop.max() - data_crop.min()
    )
    colors_rgb = np.vstack((colors, colors, colors))

    data_crop_pnts = np.vstack(
        (xpc.T.flatten(), ypc.T.flatten(), data_crop.T.flatten())
    )
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(data_crop_pnts.T)

    pnt_idx = np.reshape(
        np.arange(crop_width * crop_height), (crop_height, crop_width)
    ).T
    tri_a = np.vstack(
        (
            pnt_idx[:-1, :-1].T.flatten(),
            pnt_idx[1:, :-1].T.flatten(),
            pnt_idx[:-1, 1:].T.flatten(),
        )
    )
    tri_b = np.vstack(
        (
            pnt_idx[1:, 1:].T.flatten(),
            pnt_idx[:-1, 1:].T.flatten(),
            pnt_idx[1:, :-1].T.flatten(),
        )
    )
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.hstack((tri_a, tri_b)).T)
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors_rgb.T)

    if args.output_png is not None:
        io.imsave(args.output_png, data_crop)

    o3d.io.write_triangle_mesh(args.output_ply, o3d_mesh)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
