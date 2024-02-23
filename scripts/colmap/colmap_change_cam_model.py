#!/usr/bin/env python3


"""
Change camera model in COLMAP model

"""

import argparse
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_path", 
    type=str, 
    required=True, 
    help="Path to the input COLMAP model"
)
parser.add_argument(
    "--output_path", 
    type=str, 
    required=True, 
    help="Path to the output COLMAP model"
)
parser.add_argument(
    "--output_cam_model",
    type=str,
    choices=[
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
        "FULL_OPENCV",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
        "OPENCV_FISHEYE",
        "FOV",
        "THIN_PRISM_FISHEYE",
    ],
    help="Camera model of the output COLMAP model",
)
parser.add_argument(
    "--fov",
    type=float,
    help="The field of view of fisheye camera used in FOV camera model.",
)
parser.add_argument(
    "--output_type",
    type=str,
    choices=["BIN", "TXT", "NVM"],
    default="TXT",
    help="Type of the output model",
)


def main(args):
    colmap_model = pycolmap.Reconstruction(args.input_path)

    # print(dir(colmap_model))

    if args.output_cam_model is not None:
        new_cameras_list = []

        for cam_id, cam in colmap_model.cameras.items():
            # All params:
            # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, sx1, sy1, omega
            k1 = k2 = p1 = p2 = k3 = k4 = k5 = k6 = sx1 = sy1 = 0
            omega = args.fov

            # Read the input parameters
            if cam.model_name == "SIMPLE_PINHOLE":
                # f, cx, cy
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
            elif cam.model_name == "PINHOLE":
                # fx, fy, cx, cy
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
            elif (
                cam.model_name == "SIMPLE_RADIAL"
                or cam.model_name == "SIMPLE_RADIAL_FISHEYE"
            ):
                # f, cx, cy, k
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
                k1 = cam.params[3]
            elif cam.model_name == "RADIAL" or cam.model_name == "RADIAL_FISHEYE":
                # f, cx, cy, k1, k2
                fx = fy = cam.params[0]
                cx = cam.params[1]
                cy = cam.params[2]
                k1 = cam.params[3]
                k2 = cam.params[4]
            elif cam.model_name == "OPENCV":
                # fx, fy, cx, cy, k1, k2, p1, p2
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
                k1 = cam.params[4]
                k2 = cam.params[5]
                p1 = cam.params[6]
                p2 = cam.params[7]
            elif cam.model_name == "FULL_OPENCV":
                # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
                k1 = cam.params[4]
                k2 = cam.params[5]
                p1 = cam.params[6]
                p2 = cam.params[7]
                k3 = cam.params[8]
                k4 = cam.params[9]
                k5 = cam.params[10]
                k6 = cam.params[11]
            elif cam.model_name == "OPENCV_FISHEYE":
                # fx, fy, cx, cy, k1, k2, k3, k4
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
                k1 = cam.params[4]
                k2 = cam.params[5]
                k3 = cam.params[6]
                k4 = cam.params[7]
            elif cam.model_name == "FOV":
                # fx, fy, cx, cy, omega
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
                omega = cam.params[4]
            elif cam.model_name == "THIN_PRISM_FISHEYE":
                # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
                fx = cam.params[0]
                fy = cam.params[1]
                cx = cam.params[2]
                cy = cam.params[3]
                k1 = cam.params[4]
                k2 = cam.params[5]
                p1 = cam.params[6]
                p2 = cam.params[7]
                k3 = cam.params[8]
                k4 = cam.params[9]
                sx1 = cam.params[10]
                sy1 = cam.params[11]

            # Create output list of parameters
            if args.output_cam_model == "SIMPLE_PINHOLE":
                # f, cx, cy
                new_params = [fx, cx, cy]
            elif args.output_cam_model == "PINHOLE":
                # fx, fy, cx, cy
                new_params = [fx, fy, cx, cy]
            elif (
                args.output_cam_model == "SIMPLE_RADIAL"
                or args.output_cam_model == "SIMPLE_RADIAL_FISHEYE"
            ):
                # f, cx, cy, k
                new_params = [fx, cx, cy, k1]
            elif (
                args.output_cam_model == "RADIAL"
                or args.output_cam_model == "RADIAL_FISHEYE"
            ):
                # f, cx, cy, k1, k2
                new_params = [fx, cx, cy, k1, k2]
            elif args.output_cam_model == "OPENCV":
                # fx, fy, cx, cy, k1, k2, p1, p2
                new_params = [fx, fy, cx, cy, k1, k2, p1, p2]
            elif args.output_cam_model == "FULL_OPENCV":
                # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                new_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
            elif args.output_cam_model == "OPENCV_FISHEYE":
                # fx, fy, cx, cy, k1, k2, k3, k4
                new_params = [fx, fy, cx, cy, k1, k2, k3, k4]
            elif args.output_cam_model == "FOV":
                # fx, fy, cx, cy, omega
                new_params = [fx, fy, cx, cy, omega]
            elif args.output_cam_model == "THIN_PRISM_FISHEYE":
                # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
                new_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]

            colmap_model.cameras[cam_id] = pycolmap.Camera(
                model=args.output_cam_model,
                width=cam.width,
                height=cam.height,
                params=new_params,
            )

    if args.output_type == "BIN":
        colmap_model.write_binary(args.output_path)
    elif args.output_type == "TXT":
        colmap_model.write_text(args.output_path)
    elif args.output_type == "NVM":
        colmap_model.export_NVM(args.output_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
