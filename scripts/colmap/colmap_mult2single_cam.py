#!/usr/bin/env python3


"""
Convert COLMAP model from multiple cameras to a single one (assuming there are 
multiple cameras in the model, but only single camera in reality) by averaging 
the camera parameters.

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
    "--output_type",
    type=str,
    required=False,
    choices=["BIN", "TXT"],
    default="BIN",
    help="Type of the output model",
)


def main(args):
    input_colmap_model = pycolmap.Reconstruction(args.input_path)
    output_colmap_model = pycolmap.Reconstruction()
    single_cam = pycolmap.Camera()

    h = w = model_name = None

    # All params:
    # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, sx1, sy1, omega

    fx = fy = cx = cy = k1 = k2 = p1 = p2 = k3 = k4 = k5 = k6 = sx1 = sy1 = omega = 0

    for cam_id, cam in input_colmap_model.cameras.items():
        if h is None:
            h = cam.height
            w = cam.width
            model_name = cam.model_name
        else:
            # check if the camera model and image size is the same
            assert h == cam.height
            assert w == cam.width
            assert model_name == cam.model_name

        # Read the input parameters
        if cam.model_name == "SIMPLE_PINHOLE":
            # f, cx, cy
            fx += cam.params[0]
            fy += cam.params[0]
            cx += cam.params[1]
            cy += cam.params[2]
        elif cam.model_name == "PINHOLE":
            # fx, fy, cx, cy
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
        elif (
            cam.model_name == "SIMPLE_RADIAL"
            or cam.model_name == "SIMPLE_RADIAL_FISHEYE"
        ):
            # f, cx, cy, k
            fx += cam.params[0]
            fy += cam.params[0]
            cx += cam.params[1]
            cy += cam.params[2]
            k1 += cam.params[3]
        elif cam.model_name == "RADIAL" or cam.model_name == "RADIAL_FISHEYE":
            # f, cx, cy, k1, k2
            fx += cam.params[0]
            fy += cam.params[0]
            cx += cam.params[1]
            cy += cam.params[2]
            k1 += cam.params[3]
            k2 += cam.params[4]
        elif cam.model_name == "OPENCV":
            # fx, fy, cx, cy, k1, k2, p1, p2
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
            k1 += cam.params[4]
            k2 += cam.params[5]
            p1 += cam.params[6]
            p2 += cam.params[7]
        elif cam.model_name == "FULL_OPENCV":
            # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
            k1 += cam.params[4]
            k2 += cam.params[5]
            p1 += cam.params[6]
            p2 += cam.params[7]
            k3 += cam.params[8]
            k4 += cam.params[9]
            k5 += cam.params[10]
            k6 += cam.params[11]
        elif cam.model_name == "OPENCV_FISHEYE":
            # fx, fy, cx, cy, k1, k2, k3, k4
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
            k1 += cam.params[4]
            k2 += cam.params[5]
            k3 += cam.params[6]
            k4 += cam.params[7]
        elif cam.model_name == "FOV":
            # fx, fy, cx, cy, omega
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
            omega += cam.params[4]
        elif cam.model_name == "THIN_PRISM_FISHEYE":
            # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
            fx += cam.params[0]
            fy += cam.params[1]
            cx += cam.params[2]
            cy += cam.params[3]
            k1 += cam.params[4]
            k2 += cam.params[5]
            p1 += cam.params[6]
            p2 += cam.params[7]
            k3 += cam.params[8]
            k4 += cam.params[9]
            sx1 += cam.params[10]
            sy1 += cam.params[11]

    fx = fx / input_colmap_model.num_cameras()
    fy = fy / input_colmap_model.num_cameras()
    cx = cx / input_colmap_model.num_cameras()
    cy = cy / input_colmap_model.num_cameras()
    k1 = k1 / input_colmap_model.num_cameras()
    k2 = k2 / input_colmap_model.num_cameras()
    k3 = k3 / input_colmap_model.num_cameras()
    k4 = k4 / input_colmap_model.num_cameras()
    p1 = p1 / input_colmap_model.num_cameras()
    p2 = p2 / input_colmap_model.num_cameras()
    sx1 = sx1 / input_colmap_model.num_cameras()
    sy1 = sy1 / input_colmap_model.num_cameras()
    omega = omega / input_colmap_model.num_cameras()

    # Create output list of parameters
    if model_name == "SIMPLE_PINHOLE":
        # f, cx, cy
        new_params = [fx, cx, cy]
    elif model_name == "PINHOLE":
        # fx, fy, cx, cy
        new_params = [fx, fy, cx, cy]
    elif model_name == "SIMPLE_RADIAL" or model_name == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k
        new_params = [fx, cx, cy, k1]
    elif model_name == "RADIAL" or model_name == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2
        new_params = [fx, cx, cy, k1, k2]
    elif model_name == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2
        new_params = [fx, fy, cx, cy, k1, k2, p1, p2]
    elif model_name == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        new_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6]
    elif model_name == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4
        new_params = [fx, fy, cx, cy, k1, k2, k3, k4]
    elif model_name == "FOV":
        # fx, fy, cx, cy, omega
        new_params = [fx, fy, cx, cy, omega]
    elif model_name == "THIN_PRISM_FISHEYE":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1
        new_params = [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1]

    output_colmap_model = pycolmap.Reconstruction()
    output_colmap_model.add_camera(
        pycolmap.Camera(model=model_name, width=w, height=h, params=new_params, id=1)
    )

    for img_id, img in input_colmap_model.images.items():
        new_image = pycolmap.Image(
            name=img.name, tvec=img.tvec, qvec=img.qvec, camera_id=1, id=img.image_id
        )
        output_colmap_model.add_image(new_image)

    for img_id, img in input_colmap_model.images.items():
        output_colmap_model.register_image(img_id)

    if args.output_type == "BIN":
        output_colmap_model.write_binary(args.output_path)
    elif args.output_type == "TXT":
        output_colmap_model.write_text(args.output_path)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
