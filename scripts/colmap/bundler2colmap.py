#!/usr/bin/env python3


"""
Convert Bundle file to COLMAP reconstruction.

WARN: At the moment converts only camera parameters (intrinsic and extrinsic),
      the 3D points from the Bundle file are ignored.

"""


import os
import math
import numpy as np
import argparse
import pycolmap
import imagesize
from tqdm import tqdm


parser = argparse.ArgumentParser(
    description="Bundler to COLMAP converter",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "bundle_model", 
    type=str, 
    help="Path to the input Bundle v0.3 model file"
)
parser.add_argument(
    "img_list", 
    type=str, 
    help="Path to the input image list (Bundle v0.3 file)"
)
parser.add_argument(
    "image_dir", 
    type=str, 
    help="Path to the directory with images"
)
parser.add_argument(
    "colmap_model", 
    type=str, 
    help="Path to the output COLMAP model dir"
)


def main(args):
    img_list = read_img_list(args.img_list)
    cameras, points = read_bundler(args.bundle_model, img_list)
    cameras = get_size(args.image_dir, cameras)
    write_colmap(args.colmap_model, cameras, points)


def read_bundler(file_path, img_list):
    f = open(file_path, "r")

    num_cams, num_points = None, None
    cam_iter = 0

    cameras = {}
    points = []

    for line in tqdm(f):
        # - skip comments
        if line.strip().startswith("#"):
            continue

        # - read number of cameras and points in the beginning of the file
        if num_cams is None:
            words = line.strip().split()
            num_cams, num_points = int(words[0]), int(words[1])
            assert num_cams == len(img_list)
            continue

        if cam_iter < num_cams:
            # - load cameras
            cam_data = {}
            words = line.split()
            cam_data["f"] = float(words[0])
            cam_data["k1"] = float(words[1])
            cam_data["k2"] = float(words[2])

            line = next(f)
            Rw0 = list(map(float, line.split()))
            line = next(f)
            Rw1 = list(map(float, line.split()))
            line = next(f)
            Rw2 = list(map(float, line.split()))

            # Bundler v0.3 follows CG (OpenGL) cam. coordinate frame convention
            # --> multiply the Y and Z axes by -1 (rotate around X by 180 degs)
            cam_data["R"] = np.array(
                [
                    [Rw0[0], Rw0[1], Rw0[2]],
                    [-Rw1[0], -Rw1[1], -Rw1[2]],
                    [-Rw2[0], -Rw2[1], -Rw2[2]],
                ]
            )

            line = next(f)
            tw = list(map(float, line.split()))
            cam_data["t"] = [tw[0], -tw[1], -tw[2]]

            cameras[img_list[cam_iter]] = cam_data
            cam_iter = cam_iter + 1
        else:
            # - load 3D points
            pnt_data = {}
            posw = line.split()
            pnt_data["pos"] = np.array([[posw[0]], [posw[1]], [posw[2]]])

            line = next(f)
            colw = line.split()
            pnt_data["color"] = np.array([[colw[0]], [colw[1]], [colw[2]]])

            line = next(f)
            # - use first map to float to extract numbers from scientific format
            #   such as 1.603e004
            pnt_data["views"] = list(map(int, (list(map(float, line.split())))))

            points.append(pnt_data)

    f.close()
    return cameras, points


def read_img_list(file_path):
    f = open(file_path, "r")

    img_list = []
    for line in f:
        img_list.append(line.strip())

    f.close()

    return img_list


def get_size(img_dir, cameras):
    for cam in cameras:
        img_path = os.path.join(img_dir, cam)
        w, h = imagesize.get(img_path)
        cameras[cam]["w"] = w
        cameras[cam]["h"] = h

    return cameras


def write_colmap(dir_path, cameras, points):
    colmap_model = pycolmap.Reconstruction()

    cam_id = 1
    for cam in cameras:
        f = cameras[cam]["f"]
        k1 = cameras[cam]["k1"]
        k2 = cameras[cam]["k2"]
        w = cameras[cam]["w"]
        h = cameras[cam]["h"]
        R = cameras[cam]["R"]
        tvec = cameras[cam]["t"]
        qvec = R2quat(R)

        if (abs(k1) < 1e-6) and (abs(k2) < 1e-6):
            colmap_cam = pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=w,
                height=h,
                params=[f, w / 2, h / 2],
                id=cam_id,
            )
        else:
            colmap_cam = pycolmap.Camera(
                model="RADIAL", width=w, height=h, params=[f, w / 2, h / 2, k1, k2]
            )

        colmap_img = pycolmap.Image(
            name=cam, keypoints=[], tvec=tvec, qvec=qvec, camera_id=cam_id, id=cam_id
        )

        colmap_model.add_camera(colmap_cam)
        colmap_model.add_image(colmap_img, True)
        colmap_model.register_image(cam_id)

        cam_id = cam_id + 1

    # TODO: add 3D points
    # for pnt in points:

    #     pnt3D = pycolmap.Point3D(
    #         xyz=np.reshape(pnt["pos"], (3,1)).astype(np.float64),
    #         color=np.reshape(pnt["color"], (3,1))
    #         )

    #     colmap_model.add_point3D(pnt3D)

    colmap_model.write(dir_path)
    colmap_model.write_text(dir_path)


def R2quat(R):
    tr = np.trace(R)

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([[w], [x], [y], [z]])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
