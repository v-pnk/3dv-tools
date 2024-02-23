#!/usr/bin/env python


"""
Convert a depth map in NumPy format to a point cloud in XYZ format. The camera
parameters are taken from a COLMAP model. The resulting point cloud is in 
the coordinate frame of the COLMAP model.
"""


import os
import argparse
import numpy as np
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument("np_depth", type=str, help="The input depth map")
parser.add_argument("xyz_pc", type=str, help="The output point cloud")
parser.add_argument("input_colmap", type=str, help="The input colmap model")


def main(args):
    output_pc = np.empty((3, 0), dtype=np.float32)

    depth_map = np.load(args.np_depth)["depth"]

    # load the data from COLMAP model
    colmap_model = pycolmap.Reconstruction(args.input_colmap)

    depth_name = os.path.basename(args.np_depth)
    depth_name = remove_postfix(depth_name, ["_depth.npz"])

    # find the image with the same name as the depth map
    colmap_image = None
    colmap_camera = None
    for img_id, img in colmap_model.images.items():
        image_name = remove_postfix(
            os.path.basename(img.name),
            [".png", ".jpg", "_rendered_no_color.png", "_rendered_color.png"],
        )

        if image_name == depth_name:
            colmap_image = img
            colmap_camera = colmap_model.cameras[img.camera_id]
            break

    assert (
        colmap_image is not None
    ), "Could not find the image with the same name as the depth map"

    T = np.eye(4)
    T[0:3, 0:3] = np.linalg.inv(quat2R(colmap_image.qvec))
    T[0:3, 3] = -colmap_image.tvec

    fx = colmap_camera.focal_length_x

    # convert the depth map to point cloud
    pc = depth2pc(depth_map, fx)
    pc = transform_pc(pc, T)
    save_xyz(pc, args.xyz_pc)


def remove_postfix(string, postfix_list):
    for postfix in postfix_list:
        if string.endswith(postfix):
            string = string[: -len(postfix)]
    return string


# Quaternion (WXYZ) to rotation matrix
def quat2R(q):
    R = np.array(
        [
            [
                1 - 2 * (q[2] * q[2] + q[3] * q[3]),
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[1] * q[2] + q[0] * q[3]),
                1 - 2 * (q[1] * q[1] + q[3] * q[3]),
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[1] * q[3] - q[0] * q[2]),
                2 * (q[2] * q[3] + q[0] * q[1]),
                1 - 2 * (q[1] * q[1] + q[2] * q[2]),
            ],
        ]
    )

    R = np.squeeze(R)

    return R


def depth2pc(depth_map, fx):
    depth_map = depth_map.squeeze()
    Y, X = np.where(depth_map > 0.0)
    num_valid = np.count_nonzero(depth_map)
    xyz = np.ones((num_valid, 3))
    xyz[:, 0] = X.astype(np.float32) - depth_map.shape[1] / 2.0
    xyz[:, 1] = Y.astype(np.float32) - depth_map.shape[0] / 2.0
    xyz[:, 2] = fx
    xyz = xyz * depth_map[Y, X][:, None] / fx

    return xyz.transpose()


def transform_pc(xyz, T):
    xyz = np.matmul(T[0:3, 0:3], xyz)
    xyz = xyz + T[0:3, 3][:, None]

    return xyz


def save_xyz(xyz, path):
    xyz = xyz.transpose()
    np.savetxt(path, xyz, fmt="%.3f")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
