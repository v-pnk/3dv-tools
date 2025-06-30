#!/usr/bin/env python3


"""
Visualize single or multiple sets of camera poses (using Open3D library)
- camera definitions can be:
  - Virtual Rephotography format - directory with .cam and .res files
    - .res file - single line - "<width> <height>"
    - .cam file - two lines
      - "<t(3)> <R(9)>"
      - "<fx/max(w,h)> <dist0> <dist1> <fy/fx> <cx/w> <1-(cy/h)>"
  - Pseudo Ground Truth format - line per camera
    - "<image_name> <qvec(4)> <tvec(3)>"
  - Nerfstudio camera trajectory format - single .json file
  - COLMAP model - directory with images and cameras files
  - stacked directories with other camera definitions inside (e.g. multiple 
    COLMAP models)
"""


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
import open3d as o3d


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cam_defs",
    type=str,
    nargs="+",
    help="Path to one or multiple files/directories of camera definition (e.g. vrephoto "
    "directory with .cam and .res files, or directory of a COLMAP model)",
)
parser.add_argument(
    "--model_path", 
    type=str, 
    help="Path to the 3D model (mesh or point cloud)"
)
parser.add_argument(
    "--cam_scale",
    default=0.1,
    type=float,
    help="Scale of the camera 3D model (in scene units), default: %(default)s",
)
parser.add_argument(
    "--coordinate_frame_scale",
    default=1.0,
    type=float,
    help="Scale of the coordinate frame visualization (in scene units), default: %(default)s",
)
parser.add_argument(
    "--cam_rand_shift",
    default=0.0,
    type=float,
    help="Shift camera poses by the given amount (in random direction) to see "
    "also overlapping cameras: %(default)s",
)
parser.add_argument(
    "--train_split_file", 
    type=str, 
    help="Train split file in HoliCity format"
)
parser.add_argument(
    "--test_split_file", 
    type=str, 
    help="Test split file in HoliCity format"
)
parser.add_argument(
    "--max_cam_dist",
    default=-1,
    type=float,
    help="Maximum distance of camera from center of the mesh (in multiplies "
    "of the length of max side of axis-aligned bounding box of the given "
    "mesh)",
)
parser.add_argument(
    "--cam_colormap",
    type=str,
    default="rainbow",
    help="Matplotlib colormap used for coloring of the cameras",
)
parser.add_argument(
    "--cam_color_file",
    type=str,
    help="File containing values which will be used for coloring of "
    'the cameras - line in format: "<image name> <value>" or '
    '"<image name> <RGB (3)>"',
)
parser.add_argument(
    "--max_color_val",
    default=1000.0,
    type=float,
    help="Clip threshold for values in cam_color_file, default: %(default)s",
)
parser.add_argument(
    "--color_val_fnc",
    default="log",
    choices=["lin", "log", "exp"],
    type=str,
    help="Function which to apply to the values in cam_color_file",
)
parser.add_argument(
    "--every_nth_cam",
    default=1,
    type=int,
    help="Visualize only every n-th camera",
)
parser.add_argument(
    "--mesh_vis_mode",
    type=str,
    choices=[
        "monochromatic",
        "texture",
        "grad+x",
        "grad-x",
        "grad+y",
        "grad-y",
        "grad+z",
        "grad-z",
        "normals",
    ],
    default="texture",
    help="Mesh coloring mode - monochromatic / texture (works also for vertex-colored meshes) / gradient along x/y/z axis / normals",
)
parser.add_argument(
    "--shading",
    type=str,
    choices=["on", "off"],
    default="off",
    help="Shade the mesh based on triangle normals",
)
parser.add_argument(
    "--show_back_face",
    type=str,
    choices=["on", "off"],
    default="off",
    help="Show back faces of the mesh",
)
parser.add_argument(
    "--background_color",
    type=float,
    default=[255 / 255, 255 / 255, 255 / 255],
    nargs="+",
    help="Background color of the visualization - default: %(default)s",
)
parser.add_argument(
    "--pairs_file", 
    type=str, 
    help="Pairs for visualization - two image names per line",
)


def main(args):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    if args.model_path is not None:
        if os.path.splitext(args.model_path)[1] in [".xyz"]:
            model_pcd = o3d.io.read_point_cloud(args.model_path, print_progress=True)
            vis.add_geometry(model_pcd)

            aabb = model_pcd.get_axis_aligned_bounding_box()
            aabb_center = aabb.get_center()
            vis.get_view_control().set_lookat(aabb_center)
            max_side = np.max(aabb.get_extent())
        elif os.path.isdir(args.model_path) and (os.path.exists(os.path.join(args.model_path, "points3D.bin")) or os.path.exists(os.path.join(args.model_path, "points3D.txt"))):
            import pycolmap

            model = pycolmap.Reconstruction(args.model_path)

            pnts3d = np.empty((len(model.points3D), 3))
            colors = 0.5*np.ones((len(model.points3D), 3))
            
            for pnt3d_idx, point3D in enumerate(model.points3D.values()):
                pnts3d[pnt3d_idx] = point3D.xyz
                colors[pnt3d_idx] = point3D.color / 255.0

            model_pcd = o3d.geometry.PointCloud()
            model_pcd.points = o3d.utility.Vector3dVector(pnts3d)
            model_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(model_pcd)

            aabb = model_pcd.get_axis_aligned_bounding_box()
            aabb_center = aabb.get_center()
            vis.get_view_control().set_lookat(aabb_center)
            max_side = np.max(aabb.get_extent())
        else:
            # Color the mesh
            if args.mesh_vis_mode == "texture":
                model_mesh = o3d.io.read_triangle_mesh(
                    args.model_path, enable_post_processing=True, print_progress=True
                )
            else:
                model_mesh = o3d.io.read_triangle_mesh(
                    args.model_path, print_progress=True
                )
                if args.mesh_vis_mode.startswith("grad"):
                    if args.mesh_vis_mode == "grad+x":
                        model_mesh_vertex = np.asarray(model_mesh.vertices)[:, 0, None]
                    if args.mesh_vis_mode == "grad-x":
                        model_mesh_vertex = -np.asarray(model_mesh.vertices)[:, 0, None]
                    if args.mesh_vis_mode == "grad+y":
                        model_mesh_vertex = np.asarray(model_mesh.vertices)[:, 1, None]
                    if args.mesh_vis_mode == "grad-y":
                        model_mesh_vertex = -np.asarray(model_mesh.vertices)[:, 1, None]
                    if args.mesh_vis_mode == "grad+z":
                        model_mesh_vertex = np.asarray(model_mesh.vertices)[:, 2, None]
                    if args.mesh_vis_mode == "grad-z":
                        model_mesh_vertex = -np.asarray(model_mesh.vertices)[:, 2, None]

                    model_mesh_vertex = 1 - (
                        model_mesh_vertex - model_mesh_vertex.min()
                    ) / (model_mesh_vertex.max() - model_mesh_vertex.min())
                    model_mesh_vertex_colors = np.tile(model_mesh_vertex, (1, 3))
                    model_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        model_mesh_vertex_colors
                    )

                elif args.mesh_vis_mode == "normals":
                    model_mesh.compute_vertex_normals(normalized=True)
                    model_mesh_vertex_normals = 0.5 * (
                        np.asarray(model_mesh.vertex_normals) + 1.0
                    )
                    model_mesh.vertex_colors = o3d.utility.Vector3dVector(
                        model_mesh_vertex_normals
                    )

            if args.shading == "on":
                model_mesh.compute_triangle_normals()

            # - check if it's a mesh or a point cloud
            if len(model_mesh.triangles) == 0:
                mesh_tmp = o3d.geometry.PointCloud()
                mesh_tmp.points = o3d.utility.Vector3dVector(model_mesh.vertices)
                model_mesh = mesh_tmp

            vis.add_geometry(model_mesh)

            aabb = model_mesh.get_axis_aligned_bounding_box()
            aabb_center = aabb.get_center()
            vis.get_view_control().set_lookat(aabb_center)
            max_side = np.max(aabb.get_extent())
            print("Mesh model axis-aligned bounding box corners:")
            print(aabb.get_print_info())

    num_dirs = 0
    cameras = {}
    if args.cam_defs is not None:
        for cam_dir_i, cam_dir in enumerate(args.cam_defs):
            cam_defs_type = get_cam_defs_type(cam_dir)

            if cam_defs_type == "vrephoto":
                cameras[cam_dir_i] = load_vrephoto(cam_dir)

            elif cam_defs_type == "pgt_posefile":
                cameras[cam_dir_i] = load_pgt_posefile(cam_dir)

            elif cam_defs_type == "colmap":
                cameras[cam_dir_i] = load_colmap(cam_dir)

            elif cam_defs_type == "ns_cam_traj":
                cameras[cam_dir_i] = load_ns_cam_traj(cam_dir)

            elif cam_defs_type == "subdir":
                args.cam_defs += [
                    os.path.join(cam_dir, dir_name) for dir_name in os.listdir(cam_dir)
                ]
                continue

            num_dirs += 1

    # - show only every n-th camera
    for cam_dir_i in cameras.keys():
        cam_i = 0
        for cam_name in list(cameras[cam_dir_i].keys()):
            if cam_i % args.every_nth_cam != 0:
                del cameras[cam_dir_i][cam_name]
            cam_i += 1

    if args.pairs_file is not None:
        pairs = parse_pairs_file(args.pairs_file)

        for cam_dir_i in cameras.keys():
            for cam_i in cameras[cam_dir_i]:
                if cameras[cam_dir_i][cam_i]["name"] in pairs:
                    for pair_i in pairs[cameras[cam_dir_i][cam_i]["name"]]:
                        if pair_i in cameras[cam_dir_i]:
                            cameras[cam_dir_i][cam_i]["pairs"] = pair_i

    # - filter cameras based on distance to the mesh model center (max_cam_dist)
    if (args.model_path is not None) and (args.max_cam_dist > 0):
        filter_if_too_far(cameras, aabb_center, max_side, args.max_cam_dist)

    # - define camera color palette used if no color file is given
    cam_palette = matplotlib.cm.get_cmap(args.cam_colormap)(
        np.linspace(0.0, 1.0, num_dirs)
    )[:, 0:3].T
    rng = np.random.default_rng(42)
    cam_palette = rng.permutation(cam_palette)

    # - load the color file if given
    if args.cam_color_file is not None:
        load_color_file(args.cam_color_file, cameras, args)

    # - visualize cameras
    cameras_vis_list = []
    for cam_dir_i in cameras.keys():
        for cam_i in cameras[cam_dir_i]:
            if not ("color" in cameras[cam_dir_i][cam_i]):
                color = cam_palette[:, cam_dir_i % cam_palette.shape[1]]
            else:
                color = cameras[cam_dir_i][cam_i]["color"]

            vis_T = cameras[cam_dir_i][cam_i]["T"]

            if args.cam_rand_shift > 0:
                rand_shift = np.random.uniform(-1.0, 1.0, 3)
                rand_shift = (
                    args.cam_rand_shift * rand_shift / np.linalg.norm(rand_shift)
                )
                vis_T[0:3, 3] = vis_T[0:3, 3] + rand_shift

            if args.cam_scale > 0:
                # - create a wireframe camera model
                cam_vis = o3d.geometry.LineSet.create_camera_visualization(
                    cameras[cam_dir_i][cam_i]["w"],
                    cameras[cam_dir_i][cam_i]["h"],
                    cameras[cam_dir_i][cam_i]["K"],
                    cameras[cam_dir_i][cam_i]["T"],
                    args.cam_scale,
                )
                cam_vis.paint_uniform_color(color)
                cameras_vis_list.append(cam_vis)
            else:
                # - show just the camera center as a point
                if len(cameras_vis_list) == 0:
                    cam_vis = o3d.geometry.PointCloud()
                    cameras_vis_list.append(cam_vis)
                cam_C = -vis_T[0:3, 0:3].T @ vis_T[0:3, 3]
                cameras_vis_list[0].points.append(cam_C)
                cameras_vis_list[0].colors.append(color)
        
    for cam_vis in cameras_vis_list:
        vis.add_geometry(cam_vis)

    # - visualize world coordinate frame
    if args.coordinate_frame_scale > 0:
        world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=args.coordinate_frame_scale
        )
        vis.add_geometry(world_cs)

    if args.show_back_face == "on":
        vis.get_render_option().mesh_show_back_face = True
    else:
        vis.get_render_option().mesh_show_back_face = False

    print("Running the visualization...")
    vis.get_render_option().background_color = np.array(args.background_color)
    vis.run()


def filter_if_too_far(cameras, aabb_center, max_side, max_cam_dist):
    mark_for_deletion = []
    for cam_dir_i in cameras.keys():
        for cam_i in cameras[cam_dir_i]:
            R = cameras[cam_dir_i][cam_i]["T"][0:3, 0:3]
            t = cameras[cam_dir_i][cam_i]["T"][0:3, 3]
            C = -R.T @ t
            C_dist = np.sqrt(np.sum((aabb_center - C) ** 2))
            if C_dist > max_cam_dist * max_side:
                mark_for_deletion.append((cam_dir_i, cam_i))

    for cam_dir_i, cam_i in mark_for_deletion:
        del cameras[cam_dir_i][cam_i]


def load_color_file(input_path, cameras, args):
    f = open(input_path, "rt")
    cmap = plt.get_cmap(args.cam_colormap)
    all_vals = []

    for line in f:
        words = line.split()

        if len(words) == 2:
            img_name = words[0]
            val = np.minimum(float(words[1]), args.max_color_val)
            if args.color_val_fnc == "log":
                val = np.log(val + 1.0)
            elif args.color_val_fnc == "exp":
                val = np.exp(val)
            for cam_dir_i in cameras.keys():
                if img_name in cameras[cam_dir_i]:
                    cameras[cam_dir_i][img_name]["val"] = val
            all_vals.append(val)

        elif len(words) == 4:
            img_name = words[0]
            color = np.reshape(np.array(list(map(float, words[1:4]))), (3, 1))
            for cam_dir_i in cameras.keys():
                if img_name in cameras[cam_dir_i]:
                    cameras[cam_dir_i][img_name]["color"] = color

    if len(all_vals) > 0:
        min_val = min(all_vals)
        max_val = max(all_vals)
        mult_val = 1.0 / (max_val - min_val)

        for cam_dir_i in cameras.keys():
            for cam_i in cameras[cam_dir_i]:
                if "val" in cameras[cam_dir_i][cam_i]:
                    rgba = cmap(mult_val * (cameras[cam_dir_i][cam_i]["val"] - min_val))
                    cameras[cam_dir_i][cam_i]["color"] = rgba[0:3]


def load_vrephoto(input_path):
    assert os.path.isdir(input_path)
    file_list = os.listdir(input_path)

    cameras_dir = {}
    for file in file_list:
        if not (file.endswith(".cam")):
            continue

        cam_file_path = os.path.join(input_path, file)
        img_name = file[:-4]
        res_file_path = os.path.join(input_path, img_name + ".res")

        w, h = parse_res_file(res_file_path)
        T, K = parse_cam_file(cam_file_path, w, h)

        cameras_dir[img_name] = {}
        cameras_dir[img_name]["w"] = w
        cameras_dir[img_name]["h"] = h
        cameras_dir[img_name]["K"] = K
        cameras_dir[img_name]["T"] = T

    return cameras_dir


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


def load_pgt_posefile(input_path):
    assert os.path.exists(input_path)
    file = open(input_path, "rt")

    # - file contains only extrinsics --> use some dummy intrinsics
    K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 400.0], [0.0, 0.0, 1.0]])
    w = 1200
    h = 800

    cameras_dir = {}
    for line in file:
        # -
        img_name, T = parse_pgt_posefile_line(line)

        cameras_dir[img_name] = {}
        cameras_dir[img_name]["w"] = w
        cameras_dir[img_name]["h"] = h
        cameras_dir[img_name]["K"] = K
        cameras_dir[img_name]["T"] = T

    return cameras_dir


def parse_pgt_posefile_line(line):
    words = line.split()
    img_name = words[0]
    qvec = np.array(list(map(float, words[1:5])))
    tvec = np.array(list(map(float, words[5:8])))
    R = quat2R(qvec)

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec

    return img_name, T


def parse_pairs_file(path):
    pairs = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line and line[0] != "#":
                words = line.split()
                img_name_1 = words[0]
                img_name_2 = words[1]

                if img_name_1 not in pairs:
                    pairs[img_name_1] = []

                pairs[img_name_1].append(img_name_2)

    return pairs


def load_colmap(input_path):
    assert os.path.isdir(input_path)
    import pycolmap

    cam_file_path = os.path.join(input_path, "cameras")
    img_file_path = os.path.join(input_path, "images")

    valid_model = (
        os.path.exists(cam_file_path + ".txt")
        and os.path.exists(img_file_path + ".txt")
    ) or (
        os.path.exists(cam_file_path + ".bin")
        and os.path.exists(img_file_path + ".bin")
    )

    assert (
        valid_model
    ), "Given COLMAP model does not contain cameras and images files: {}".format(
        input_path
    )

    model = pycolmap.Reconstruction()
    model.read(input_path)

    cameras_dir = {}
    for img in model.images.values():
        img_name = img.name
        cam = model.cameras[img.camera_id]
        w = cam.width
        h = cam.height
        K = cam.calibration_matrix()

        T = np.eye(4)
        T[0:3, 0:3] = img.rotation_matrix()
        T[0:3, 3] = img.tvec

        cameras_dir[img_name] = {}
        cameras_dir[img_name]["w"] = w
        cameras_dir[img_name]["h"] = h
        cameras_dir[img_name]["K"] = K
        cameras_dir[img_name]["T"] = T

    return cameras_dir


def load_ns_cam_traj(input_path):
    assert os.path.isfile(input_path)
    import json

    with open(input_path, "rt") as f:
        json_dict = json.load(f)

    w = json_dict["render_width"]
    h = json_dict["render_height"]
    cam_path = json_dict["camera_path"]

    cameras_dir = {}
    for cam_i, cam in enumerate(cam_path):
        fov = cam["fov"]
        foclen = w / (2.0 * np.tan(np.radians(fov) / 2))

        T_c2w = np.array(cam["camera_to_world"])

        K = np.array([[foclen, 0.0, w / 2.0], [0.0, foclen, h / 2.0], [0.0, 0.0, 1.0]])
        T = np.linalg.inv(T_c2w)

        cameras_dir[cam_i] = {}
        cameras_dir[cam_i]["w"] = w
        cameras_dir[cam_i]["h"] = h
        cameras_dir[cam_i]["K"] = K
        cameras_dir[cam_i]["T"] = T

    return cameras_dir


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


def get_cam_defs_type(cam_defs_path):
    if os.path.isfile(cam_defs_path):
        if cam_defs_path.endswith(".txt"):
            return "pgt_posefile"
        elif cam_defs_path.endswith(".json"):
            return "ns_cam_traj"
    elif os.path.isdir(cam_defs_path):
        file_list = os.listdir(cam_defs_path)
        if "cameras.txt" in file_list and "images.txt" in file_list:
            return "colmap"
        elif "cameras.bin" in file_list and "images.bin" in file_list:
            return "colmap"
        elif file_list[0].endswith(".cam") or file_list[0].endswith(".res"):
            return "vrephoto"
        elif os.path.isdir(os.path.join(cam_defs_path, file_list[0])):
            return "subdir"
        else:
            raise Exception("Unknown camera definition type: {}".format(cam_defs_path))
    else:
        raise Exception("Unknown camera definition type: {}".format(cam_defs_path))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
