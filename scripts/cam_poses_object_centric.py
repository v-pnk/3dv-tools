#!/usr/bin/env python3


"""
Generate camera poses around a given mesh, which can be later used for 
rendering. The camera poses are generated in a spherical manner around the
given mesh.

"""


import os
import math
import argparse

import numpy as np
import open3d as o3d
import sqlite3


parser = argparse.ArgumentParser(description="")
parser.add_argument("mesh_path", type=str, help="Path to the mesh file")
parser.add_argument(
    "out_colmap_model",
    type=str,
    help="Path to the directory for the output COLMAP model",
)
parser.add_argument(
    "--rendering_suffix",
    type=str,
    default="_rendered_color.png",
    help="Suffix of rendered images (for COLMAP model - default: %(default)s)",
)
parser.add_argument(
    "--colmap_database",
    type=str,
    help="Path to COLMAP database (is used to get image IDs)",
)
parser.add_argument(
    "--render_config_file",
    type=str,
    help="YAML config with the radius, azimuth and elevation definitions",
)
parser.add_argument(
    "--azimuth_samples",
    type=int,
    default=36,
    help="Number of samples in the azimuth axis (on 360 degs) - default: %(default)s",
)
parser.add_argument(
    "--elevation_samples",
    type=int,
    default=8,
    help="Number of samples in the elevation (on 180 degs) - default: %(default)s",
)
parser.add_argument(
    "--radius_mult",
    type=float,
    nargs="+",
    default=[3.0],
    help="Multiplier of the radius estimated from the bounding box of the given mesh - default: %(default)s",
)
parser.add_argument(
    "--vert_axis",
    type=float,
    nargs="+",
    default=[0.0, 0.0, 1.0],
    help="Vertical axis  - default: %(default)s",
)
parser.add_argument(
    "--center_shift",
    type=float,
    nargs="+",
    default=[0.0, 0.0, 0.0],
    help="Translation vector to shift the center of the rings - default: %(default)s",
)
parser.add_argument(
    "--img_width", 
    type=int, 
    default=800, 
    help="Image width - default: %(default)s"
)
parser.add_argument(
    "--img_height", 
    type=int, 
    default=800, 
    help="Image height - default: %(default)s"
)
parser.add_argument(
    "--cam_foclen", 
    type=float, 
    default=800, 
    help="Image height - default: %(default)s"
)
parser.add_argument(
    "--elevation_lims",
    type=float,
    nargs="+",
    default=[-10.0, 60.0],
    help="Limits of the elevation angle (degrees) - default [0, 90] (= upper halfsphere) - default: %(default)s",
)
parser.add_argument(
    "--azimuth_lims",
    type=float,
    nargs="+",
    default=[0, 360.0],
    help="Limits of the elevation angle (degrees) - default [0, 90] (= upper halfsphere) - default: %(default)s",
)
parser.add_argument(
    "--dont_visualize",
    action="store_true",
    help="Turns off the visualization of camera poses.",
)
parser.add_argument(
    "--rel_cam_scale",
    default=0.05,
    type=float,
    help="Scale of the camera 3D model relative to the length of longest side of the bounding box of the mesh model",
)
parser.add_argument(
    "--vis_cam_color",
    default=[66 / 255, 159 / 255, 255 / 255],
    type=float,
    nargs="+",
    help="Color of the visualized camera poses - three values (R,G,B) in [0,1] range",
)
parser.add_argument(
    "--photo_mode",
    action="store_true",
    help="Do not show coordinate frames, only the mesh and camera frustums.",
)
parser.add_argument(
    "--mesh_color_mode",
    type=str,
    default="z",
    choices=["z", "shaded"],
    help="Coloring mode of the mesh - default: %(default)s",
)
parser.add_argument(
    "--mesh_color",
    type=float,
    default=[66 / 255, 159 / 255, 255 / 255],
    nargs="+",
    help="Color of the mesh - default: %(default)s",
)
parser.add_argument(
    "--background_color",
    type=float,
    default=[255 / 255, 255 / 255, 255 / 255],
    nargs="+",
    help="Background color of the visualization - default: %(default)s",
)


def main(args):
    assert os.path.exists(
        args.mesh_path
    ), "The given mesh file does not exist: {}".format(args.mesh_path)

    if args.render_config_file is not None:
        load_render_config(args.render_config_file, args)

    assert (
        len(args.vert_axis) == 3
    ), "The given vertical axis must have 3 dimensions, but has {}.".format(
        len(args.vert_axis)
    )
    assert (
        len(args.center_shift) == 3
    ), "The given center shift must have 3 dimensions, but has {}.".format(
        len(args.center_shift)
    )

    assert (
        len(args.elevation_lims) == 2
    ), "The elevation limits parameter must be a list of 2 elements"
    assert (
        len(args.azimuth_lims) == 2
    ), "The azimuth limits parameter must be a list of 2 elements"
    assert (
        args.elevation_lims[0] <= args.elevation_lims[1]
    ), "The first element of elevation limits marks minimum elevation and the second maximum elevation, therefore (min <= max) has to hold"

    scene = o3d.io.read_triangle_model(args.mesh_path, print_progress=True)

    mesh = o3d.geometry.TriangleMesh()
    for mesh_i in scene.meshes:
        mesh += mesh_i.mesh

    print(len(mesh.vertices))

    bbox = mesh.get_oriented_bounding_box()

    center = np.reshape(bbox.get_center(), (3, 1))
    center_shift = np.reshape(args.center_shift, (3, 1))
    center = center + center_shift

    bbox_max_side = max(bbox.get_max_bound())

    radius = np.array(args.radius_mult) * 0.5 * bbox_max_side
    vert_axis = np.reshape(args.vert_axis, (3, 1))
    azimuth_arr = np.linspace(
        args.azimuth_lims[0], args.azimuth_lims[1], args.azimuth_samples + 1
    )[:-1]

    elevation_arr = np.linspace(
        args.elevation_lims[0], args.elevation_lims[1], args.elevation_samples
    )
    if math.isclose(args.elevation_lims[0], -90):
        elevation_arr = elevation_arr[1:]
    if math.isclose(args.elevation_lims[1], 90):
        elevation_arr = elevation_arr[:-1]
    if args.elevation_samples == 1:
        elevation_arr = np.array([0.0])

    azimuth_mesh, elevation_mesh, radius_mesh = np.meshgrid(
        azimuth_arr, elevation_arr, radius
    )
    azimuth_mesh = azimuth_mesh.flatten()
    elevation_mesh = elevation_mesh.flatten()
    radius_mesh = radius_mesh.flatten()

    # add the single top and bottom view
    if math.isclose(args.elevation_lims[0], -90):
        azimuth_mesh = np.append(azimuth_mesh, [0.0] * radius.size)
        elevation_mesh = np.append(elevation_mesh, [-90.0] * radius.size)
        radius_mesh = np.append(radius_mesh, radius)
    if math.isclose(args.elevation_lims[1], 90):
        azimuth_mesh = np.append(azimuth_mesh, [0.0] * radius.size)
        elevation_mesh = np.append(elevation_mesh, [90.0] * radius.size)
        radius_mesh = np.append(radius_mesh, radius)

    azimuth_mesh = np.radians(azimuth_mesh)
    elevation_mesh = np.radians(elevation_mesh)

    # The basic camera coordinate frame
    # - CG convention (Z points behind cam, Y points down)
    T_init_CG = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

    T_center = np.eye(4)
    T_center[0:3, 3] = center.flatten()

    T_vert_axis = np.eye(4)
    T_vert_axis[0:3, 0:3] = vert_axis2R(vert_axis)

    if not (args.dont_visualize):
        K = np.array(
            [
                [args.cam_foclen, 0.0, 0.5 * args.img_width],
                [0.0, args.cam_foclen, 0.5 * args.img_height],
                [0.0, 0.0, 1.0],
            ]
        )

        vis = o3d.visualization.Visualizer()
        vis.create_window()

    if args.out_colmap_model is not None:
        if args.colmap_database is not None:
            img_name_to_img_id = getImgNameToImgIdMap(args.colmap_database)
        else:
            print(
                "WARN: The generated COLMAP model image IDs will probably not align to the IDs in any model generated by COLMAP."
            )
            print(
                "- Create first a model with a database in COLMAP and provide the database to this script to fix the issue."
            )

        colmap_images_path = os.path.join(args.out_colmap_model, "images.txt")
        colmap_cameras_path = os.path.join(args.out_colmap_model, "cameras.txt")
        colmap_points3D_path = os.path.join(args.out_colmap_model, "points3D.txt")

        f_points3D = open(colmap_points3D_path, "w")
        f_points3D.close()

        f_cameras = open(colmap_cameras_path, "w")
        f_cameras.write("# Camera list with one line of data per camera:\n")
        f_cameras.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f_cameras.write("# Number of cameras: 1\n")
        f_cameras.write(
            "1 PINHOLE {} {} {} {} {} {}".format(
                args.img_width,
                args.img_height,
                args.cam_foclen,
                args.cam_foclen,
                args.img_width / 2.0,
                args.img_height / 2.0,
            )
        )
        f_cameras.close()

        f_images = open(colmap_images_path, "w")
        f_images.write("# Image list with two lines of data per image:\n")
        f_images.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f_images.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f_images.write(
            "# Number of images: {}, mean observations per image: 0.0\n".format(
                azimuth_mesh.size
            )
        )

    fake_id = 0
    for az, el, r in zip(azimuth_mesh, elevation_mesh, radius_mesh):
        fake_id += 1

        T_azimuth = np.array(
            [
                [np.cos(az), -np.sin(az), 0, 0],
                [np.sin(az), np.cos(az), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

        T_elevation = np.array(
            [
                [np.cos(-el), 0, np.sin(-el), 0],
                [0, 1, 0, 0],
                [-np.sin(-el), 0, np.cos(-el), 0],
                [0, 0, 0, 1],
            ]
        )

        T_radius = np.eye(4)
        T_radius[0, 3] = r

        T_end = T_center @ T_vert_axis @ T_azimuth @ T_elevation @ T_radius @ T_init_CG
        T_out = np.linalg.inv(T_end)
        T_colmap = T_out

        img_name = "sphere_r{:.0f}_a{:.0f}_e{:.0f}".format(
            r, np.degrees(az), np.degrees(el)
        )

        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        R_colmap = T_colmap[0:3, 0:3]
        t_colmap = T_colmap[0:3, 3]
        q_colmap = R2quat(R_colmap).flatten()
        full_img_name = img_name + args.rendering_suffix
        if args.colmap_database is not None:
            img_id = img_name_to_img_id[full_img_name]
        else:
            img_id = fake_id
        f_images.write(
            "{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} 1 {}\n\n".format(
                img_id,
                q_colmap[0],
                q_colmap[1],
                q_colmap[2],
                q_colmap[3],
                t_colmap[0],
                t_colmap[1],
                t_colmap[2],
                full_img_name,
            )
        )

        if not (args.dont_visualize):
            if not (args.photo_mode):
                T_CS_cam_scale = np.eye(4)
                T_CS_cam_scale[0:3, 0:3] = np.diag(
                    np.tile(args.rel_cam_scale * bbox_max_side, 3)
                )
                CS_cam_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
                CS_cam_mesh.transform(T_end @ T_CS_cam_scale)
                vis.add_geometry(CS_cam_mesh)

                T_mesh_align_scale = np.array(
                    [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 3, 0], [0, 0, 0, 1]]
                )
                T_mesh_align = T_center @ T_vert_axis @ T_mesh_align_scale
                CS_mesh_align = o3d.geometry.TriangleMesh.create_coordinate_frame()
                CS_mesh_align.transform(T_mesh_align)
                vis.add_geometry(CS_mesh_align)

            cam_vis = o3d.geometry.LineSet.create_camera_visualization(
                args.img_width,
                args.img_height,
                K,
                T_out,
                args.rel_cam_scale * bbox_max_side,
            )
            cam_vis.paint_uniform_color(np.array(args.vis_cam_color).T)
            vis.add_geometry(cam_vis)

    if args.out_colmap_model is not None:
        f_images.close()

    if not (args.dont_visualize):
        if args.mesh_color_mode == "z":
            mesh_pnts = (
                np.linalg.inv(T_vert_axis[0:3, 0:3]) @ np.asarray(mesh.vertices).T
            )
            mesh_z = mesh_pnts[2, :]
            mesh_z = (mesh_z - np.min(mesh_z)) / (np.max(mesh_z) - np.min(mesh_z))
            mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(mesh_z, (3, 1)).T)

            vis.add_geometry(mesh)
        elif args.mesh_color_mode == "shaded":
            mesh.compute_vertex_normals()
            mesh.compute_triangle_normals()
            mesh.paint_uniform_color(np.array(args.mesh_color).T)

            vis.add_geometry(mesh)

        if not (args.photo_mode):
            CS_world_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            CS_world_mesh.scale(1.5, np.zeros((3, 1)))
            vis.add_geometry(CS_world_mesh)

            bbox_lineset = bbox2lineset(bbox)
            bbox_lineset.paint_uniform_color(np.array([[1.0], [0.0], [0.0]]))
            vis.add_geometry(bbox_lineset)

        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().background_color = np.array(args.background_color)

        vis.run()


def vert_axis2R(vect_axis):
    vec_1 = np.array([[0], [0], [1]])
    vec_2 = vect_axis / np.sqrt(np.sum(vect_axis * vect_axis))

    # - if the vectors are same, cross product is zero vector and its norm
    #   produces NaN values
    # if np.math.isclose(vec_1, vec_2).all():
    if (np.abs(vec_1 - vec_2) < 1e-6).all():
        # - define two random vectors in case one of them would be parallel to
        #   input vectors
        rand_vect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).T
        # - choose the one which is more perpendicular to the input vectors
        # rand_vect = rand_vect[:, np.argmin(vec_1.T @ rand_vect, axis = 1)]
        # - argmin with axis specified is not implemented in numba
        dot_prod = vec_1.T @ rand_vect
        if dot_prod[0, 0] < dot_prod[0, 1]:
            rand_vect = rand_vect[:, 0]
        else:
            rand_vect = rand_vect[:, 1]
        # - compute the axis as pseudo random vector perpendicualar to the
        #   input vector
        axis = np.reshape(np.cross(vec_1.flatten(), rand_vect.flatten()), (3, 1))
        # - angle is zero
        ang = 0.0
    else:
        cross_prod = np.cross(vec_1.flatten(), vec_2.flatten())
        dot_prod = np.dot(vec_1.flatten(), vec_2.flatten())
        axis = np.reshape(cross_prod, (3, 1))
        axis = axis / np.sqrt(np.sum(axis * axis))

        # cos(ang) = (u . v) / (|u| * |v|) = (u . v) / (1 * 1) = u . v
        # ang_cos = dot_prod / (vect_len(vec_1)*vect_len(vec_2))
        ang_cos = dot_prod
        # sin(ang) = |u x v| / (|u| * |v|) = |u x v| / (1 * 1) = |u x v|
        # ang_sin = vect_len(cross_prod) / (vect_len(vec_1)*vect_len(vec_2))
        ang_sin = np.sqrt(np.sum(cross_prod * cross_prod))
        # use atan2 to compute the angle
        ang = math.atan2(ang_sin, ang_cos)

    skew_sym_axis = np.array(
        [
            [0, -axis[2, 0], axis[1, 0]],
            [axis[2, 0], 0, -axis[0, 0]],
            [-axis[1, 0], axis[0, 0], 0],
        ]
    )

    # - Rodrigues' rotation formula
    # R = I + sin(ang)*[v] + (1-cos(ang))*[v]^2
    return (
        np.eye(3)
        + math.sin(ang) * skew_sym_axis
        + (1 - math.cos(ang)) * (skew_sym_axis @ skew_sym_axis)
    )


def bbox2lineset(bbox):
    bbox_corners = bbox.get_box_points()
    # - define the box lines by corner indices
    lines = np.array(
        [
            [0, 2],
            [0, 3],
            [2, 5],
            [3, 5],
            [1, 6],
            [1, 7],
            [6, 4],
            [7, 4],
            [0, 1],
            [2, 7],
            [3, 6],
            [5, 4],
        ]
    )
    lineset = o3d.geometry.LineSet()
    lineset.points = bbox_corners
    lineset.lines = o3d.utility.Vector2iVector(lines)

    return lineset


def R2quat(R, order="WXYZ"):
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

    if order == "WXYZ":
        return np.array([[w], [x], [y], [z]])
    else:
        return np.array([[x], [y], [z], [w]])


# Code taken from uzh-rpg/colmap_utils repository
# https://github.com/uzh-rpg/colmap_utils/blob/master/utils/colmap_utils.py
def getImgNameToImgIdMap(database):
    connection = sqlite3.connect(database)
    cursor = connection.cursor()

    img_nm_to_id = {}
    cursor.execute("SELECT name, image_id FROM images;")
    for row in cursor:
        img_nm_to_id[row[0]] = row[1]

    cursor.close()
    connection.close()

    return img_nm_to_id


def load_render_config(file_path, args):
    import yaml

    with open(file_path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    if "radius_mult" in data:
        args.radius_mult = data["radius_mult"]
    if "center_shift" in data:
        args.center_shift = data["center_shift"]
    if "vert_axis" in data:
        args.vert_axis = data["vert_axis"]
    if "elevation_samples" in data:
        args.elevation_samples = data["elevation_samples"]
    if "azimuth_samples" in data:
        args.azimuth_samples = data["azimuth_samples"]
    if "elevation_lims" in data:
        args.elevation_lims = data["elevation_lims"]
    if "azimuth_lims" in data:
        args.azimuth_lims = data["azimuth_lims"]
    if "img_width" in data:
        args.img_width = data["img_width"]
    if "img_height" in data:
        args.img_height = data["img_height"]
    if "cam_foclen" in data:
        args.cam_foclen = data["cam_foclen"]


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
