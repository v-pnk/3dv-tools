#!/usr/bin/env python3


import os
import argparse

import numpy as np
import open3d as o3d
import PIL
from tqdm import tqdm
import pycolmap


parser = argparse.ArgumentParser(description="Render images from 3D model")
parser.add_argument(
    "--model", 
    type=str, 
    required=True, 
    help="Path to the 3D model"
)
parser.add_argument(
    "--colmap_model",
    type=str,
    required=False,
    help="Path to the colmap model (for camera defitions)",
)
parser.add_argument(
    "--vrephoto_dir",
    type=str,
    required=False,
    help="Path to the directory with camera definitions in vrephoto format (pairs of .cam and .res files)",
)
parser.add_argument(
    "--output_dir", 
    type=str, 
    required=True, 
    help="Path to the output directory"
)
parser.add_argument(
    "--use_color", 
    action="store_true", 
    help="Flag for color rendering"
)
parser.add_argument(
    "--gen_masks",
    action="store_true",
    help="Generate masks (in Instant-NGP format) - background is black",
)
parser.add_argument(
    "--lit",
    action="store_true",
    help="Flag to turn on lit shader for color rendering - lit is used defaultly for uncolored rendering",
)
parser.add_argument(
    "--keep_color",
    action="store_true",
    help="Apart from textures, keep also the colors of the mesh",
)
parser.add_argument(
    "--no_suffix",
    action="store_true",
    help="Do not add suffix to the output file names",
)
parser.add_argument(
    "--only_black_frames",
    action="store_true",
    help="Check output dir. and render only if the output does not exist or the image is all black",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=1.0,
    help="Check output dir. and render only if the output does not exist or the image is all black",
)
args = parser.parse_args()


def main(args):
    assert os.path.isfile(args.model)
    assert os.path.isdir(args.output_dir)

    valid_cam_info = (
        (args.colmap_model is not None) and os.path.isdir(args.colmap_model)
    ) or ((args.vrephoto_dir is not None) and os.path.isdir(args.vrephoto_dir))
    assert (
        valid_cam_info
    ), "No valid camera informations passed to the script - specify valid colmap_model or vrephoto_dir"

    # Load the mesh
    print("Loading the mesh")
    mesh = o3d.io.read_triangle_model(args.model, True)

    # Load the images
    print("Loading the images and cameras")
    cam_list = []

    if args.colmap_model is not None:
        model = pycolmap.Reconstruction(args.colmap_model)

        for img in model.images.values():
            qvec = img.qvec
            tvec = img.tvec
            cam = model.cameras[img.camera_id]
            if len(cam.focal_length_idxs()) == 1:
                fx = cam.focal_length
                fy = cam.focal_length
            else:
                fx = cam.focal_length_x
                fy = cam.focal_length_y

            cx = cam.principal_point_x
            cy = cam.principal_point_y

            w = cam.width
            h = cam.height

            K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

            R = qvec2rotmat(qvec)
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec

            basename = os.path.splitext(img.name)[0]
            cam_list.append({"basename": basename, "K": K, "T": T, "w": w, "h": h})

    elif args.vrephoto_dir is not None:
        file_list = os.listdir(args.vrephoto_dir)
        for file in file_list:
            if not (file.endswith(".cam")):
                continue

            cam_file_path = os.path.join(args.vrephoto_dir, file)
            res_file_path = os.path.join(args.vrephoto_dir, file[:-4] + ".res")

            w, h = parse_res_file(res_file_path)
            T, K = parse_cam_file(cam_file_path, w, h)

            basename = os.path.splitext(file)[0]

            cam_list.append({"basename": basename, "K": K, "T": T, "w": w, "h": h})

    # - all possible Open3D renderer shaders found in
    #   Open3D/cpp/open3d/visualization/gui/Materials/ directory
    if args.use_color:
        for iter in range(len(mesh.materials)):
            if args.lit:
                mesh.materials[iter].shader = "defaultLit"
            else:
                mesh.materials[iter].shader = "defaultUnlit"

            # - the original colors make the textures too dark - set to white
            if not (args.keep_color):
                mesh.materials[iter].base_color = [1.0, 1.0, 1.0, 1.0]

    else:
        # mesh.paint_uniform_color(np.array([[0.5],[0.5],[0.5]]))
        for mesh_i in range(len(mesh.meshes)):
            mesh.meshes[mesh_i].mesh.paint_uniform_color(
                np.array([[0.5], [0.5], [0.5]])
            )
        for mat_i in range(len(mesh.materials)):
            mesh.materials[mat_i].shader = "defaultLit"
            mesh.materials[mat_i].albedo_img = None

    for cam in tqdm(cam_list):
        if args.no_suffix:
            output_path = os.path.join(
                args.output_dir, "{}.png".format(cam["basename"].replace("/", "_"))
            )
        elif args.use_color:
            output_path = os.path.join(
                args.output_dir,
                "{}_rendered_color.png".format(cam["basename"].replace("/", "_")),
            )
        else:
            output_path = os.path.join(
                args.output_dir,
                "{}_rendered_no_color.png".format(cam["basename"].replace("/", "_")),
            )

        if args.only_black_frames and os.path.exists(output_path):
            out_img = np.asarray(PIL.Image.open(output_path))
            if not (np.all(out_img <= 1)):
                # the output exists and is not all-black frame
                continue
            else:
                print("rerendering: {}".format(os.path.basename(output_path)))

        T = cam["T"]
        K = cam["K"]
        w, h = cam["w"], cam["h"]

        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        renderer.scene.add_model("Scene mesh", mesh)

        renderer.setup_camera(K, T, w, h)

        renderer.scene.view.set_antialiasing(True)
        renderer.scene.view.set_sample_count(8)  # MSAA

        light_name_list = []

        # - setup lighting
        if args.use_color:
            renderer.scene.scene.enable_sun_light(False)
            # renderer.scene.scene.enable_sun_light(True)
        else:
            # ## Adds Raymond lights. Code taken from PyRender code at:
            # # https://github.com/mmatl/pyrender/blob/dd6dbd895aada77f33975cedaad039ac58811ea4/pyrender/viewer.py
            thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
            phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

            for phi, theta in zip(phis, thetas):
                xp = np.sin(theta) * np.cos(phi)
                yp = np.sin(theta) * np.sin(phi)
                zp = np.cos(theta)

                z = np.array([xp, yp, zp])
                z = z / np.linalg.norm(z)
                x = np.array([-z[1], z[0], 0.0])
                if np.linalg.norm(x) == 0:
                    x = np.array([1.0, 0.0, 0.0])
                x = x / np.linalg.norm(x)
                y = np.cross(z, x)

                matrix = np.eye(4)
                matrix[:3, :3] = np.c_[x, y, z]

                light_T = T @ matrix

                # - z axis corresponds to the light direction
                light_dir = light_T[0:3, 2]

                # - add_directional_light(name, color, direction, intensity, cast_shadows)
                renderer.scene.scene.add_directional_light(
                    "raymond_{:.1f}_{:.1f}".format(phi, theta),
                    np.ones((3, 1)),
                    np.reshape(light_dir, (3, 1)),
                    1000.0,
                    True,
                )
                light_name_list.append("raymond_{:.1f}_{:.1f}".format(phi, theta))

        color = np.array(renderer.render_to_image())

        # - adjust gamma of the rendered image
        color = (((color / 255.0) ** args.gamma) * 255.0).astype(np.uint8)

        depth = np.array(renderer.render_to_depth_image(True))
        depth[np.isinf(depth)] = 0.0

        img_rendering = PIL.Image.fromarray(color)
        img_rendering.save(output_path)
        np.savez_compressed(
            os.path.join(
                args.output_dir,
                "{}_depth.npz".format(cam["basename"].replace("/", "_")),
            ),
            depth=depth.astype(np.float16),
        )

        if args.gen_masks:
            mask = np.isclose(depth, 0.0).astype(np.uint8) * 255
            img_depth = PIL.Image.fromarray(mask)
            img_dir = os.path.dirname(output_path)
            img_name = os.path.basename(output_path)
            mask_path = os.path.join(img_dir, "dynamic_mask_" + img_name)
            img_depth.save(mask_path)

        # - remove all lights from the scene
        for light_name in light_name_list:
            renderer.scene.scene.remove_light(light_name)


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def parse_cam_file(path, w, h):
    R = np.eye(3)
    t = np.zeros((3, 1))
    T = np.eye(4)

    f = open(path, "r")
    line1 = f.readline()
    line2 = f.readline()

    (
        t[0],
        t[1],
        t[2],
        R[0, 0],
        R[0, 1],
        R[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
    ) = map(float, line1.split())
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()

    f_norm, _, _, aspect, cx_w, hcy_h = map(float, line2.split())

    fx = f_norm * np.float32(max(w, h))
    fy = aspect * fx
    cx = cx_w * w
    cy = h - (hcy_h * h)
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    f.close()

    return T, K


def parse_res_file(path):
    f = open(path, "r")
    line = f.readline()
    words = line.split()
    w = int(words[0])
    h = int(words[1])
    f.close()

    return w, h


if __name__ == "__main__":
    main(args)
