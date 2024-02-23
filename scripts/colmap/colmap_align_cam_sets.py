#!/usr/bin/env python3


"""
Align two camera sets with ICP (use camera centers as point clouds)

"""


import os
import numpy as np
import math
import argparse
import open3d as o3d
import copy
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument(
    "desired_colmap_model",
    type=str,
    help="Path to the COLMAP model with desired camera poses",
)
parser.add_argument(
    "given_colmap_model",
    type=str,
    help="Path to the COLMAP model which will be transformed",
)
parser.add_argument(
    "transformed_colmap_model",
    type=str,
    help="Path to the directory where the transformed COLMAP model (originally the given_colmap_model) will be written",
)

parser.add_argument(
    "--n_icp_tries",
    type=int,
    default=1,
    help="Number of tries of ICP - with random initializations, default: %(default)s",
)
parser.add_argument(
    "--icp_max_corr_dist",
    type=float,
    default=1000.0,
    help="Maximum correspondence distance (ICP param.), default: %(default)s",
)

parser.add_argument(
    "--use_convex_hull",
    action="store_true",
    help="Use just convex hulls of camera centers as inputs to ICP",
)
parser.add_argument(
    "--try_symmetries",
    action="store_true",
    help="Tries to apply a series of 90 degree rotations around principal axes on the ICP result",
)

parser.add_argument(
    "--vis_pos_align",
    action="store_true",
    help="Visualize the resulting alignment of camera positions",
)


def main(args):
    print("COLMAP model alignment")
    assert os.path.isdir(args.desired_colmap_model)
    assert os.path.isdir(args.given_colmap_model)
    assert os.path.isdir(args.transformed_colmap_model)

    # - parse the models and get positions of the cameras
    print("- reading the desired COLMAP model")
    model_desired = pycolmap.Reconstruction(args.desired_colmap_model)
    desired_pos_arr = colmap2cam_pc(model_desired)
    desired_pos_arr_orig = desired_pos_arr.copy()

    print("- reading the given COLMAP model")
    model_given = pycolmap.Reconstruction(args.given_colmap_model)
    given_pos_arr = colmap2cam_pc(model_given)
    given_pos_arr_orig = given_pos_arr.copy()

    # - check if the two sets contain tha same number of camera positions
    assert (
        desired_pos_arr.shape == given_pos_arr.shape
    ), "ERROR: desired poses: {}, given poses: {}".format(
        desired_pos_arr.shape, given_pos_arr.shape
    )

    # - find maximum distance between two cameras within single set and compute
    #   scale ratio between the sets
    print("- estimating the scale factor between the sets")
    max_dist_desired = max_dist(desired_pos_arr)
    max_dist_given = max_dist(given_pos_arr)
    s = max_dist_desired / max_dist_given
    print("  - scale factor s = {:.3f}".format(s))

    # - scale the second set
    given_pos_arr = s * given_pos_arr

    desired_pos_pc = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(desired_pos_arr.T)
    )
    desired_pos_pc.paint_uniform_color(np.array([[1], [0], [0]]))
    given_pos_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(given_pos_arr.T))
    given_pos_pc.paint_uniform_color(np.array([[0], [1], [0]]))

    if args.use_convex_hull:
        print("- generating convex hull")
        desired_pos_ch, _ = desired_pos_pc.compute_convex_hull()
        desired_pos_pc = o3d.geometry.PointCloud(desired_pos_ch.vertices)
        desired_pos_pc.paint_uniform_color(np.array([[1], [0], [0]]))

        given_pos_ch, _ = given_pos_pc.compute_convex_hull()
        given_pos_pc = o3d.geometry.PointCloud(given_pos_ch.vertices)
        given_pos_pc.paint_uniform_color(np.array([[0], [1], [0]]))

    # - center the sets
    print("- centering the camera sets")
    ct = desired_pos_pc.get_center() - given_pos_pc.get_center()
    given_pos_pc.translate(ct)

    print("- preparing ICP")
    icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1.000000e-10, relative_rmse=1.000000e-10, max_iteration=200
    )

    rmse_list = np.array([])
    fitness_list = np.array([])
    T_list = np.empty((4, 4, 0))

    print("- running ICP")
    for init_i in range(args.n_icp_tries):
        # Try first the original transformation
        T_init = np.eye(4)
        if init_i > 1:
            rand_rot_axis = np.random.uniform(-1, 1, (3, 1))
            rand_rot_axis = rand_rot_axis / np.sqrt(np.sum(rand_rot_axis**2))
            rand_rot_ang = np.random.uniform(-np.pi, np.pi)

            rand_R = axis_ang2R(rand_rot_axis, rand_rot_ang)

            T_init[0:3, 0:3] = rand_R

        reg_result = o3d.pipelines.registration.registration_icp(
            given_pos_pc,
            desired_pos_pc,
            max_correspondence_distance=args.icp_max_corr_dist,
            criteria=icp_criteria,
            init=T_init,
        )

        rmse_list = np.append(rmse_list, reg_result.inlier_rmse)
        fitness_list = np.append(fitness_list, reg_result.fitness)
        # T_est = reg_result.transformation @ T_init
        T_est = reg_result.transformation
        T_list = np.append(T_list, np.reshape(T_est, (4, 4, 1)), axis=2)

        print("  - Test no. {}".format(init_i))
        print("    - init. T = ")
        print(T_init)
        print("    - Fitness = {}".format(reg_result.fitness))
        print("    - Inlier RMSE = {}\n".format(reg_result.inlier_rmse))

    best_i = np.argmin(rmse_list)
    T_ICP = T_list[:, :, best_i]

    if args.try_symmetries:
        cov_mat = np.cov(given_pos_arr)
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        # Make the basis right-handed
        if np.linalg.det(eig_vecs) < 0:
            eig_vecs[0:3, 0] = -eig_vecs[0:3, 0]

        T_R_align = np.eye(4)
        T_R_align[0:3, 0:3] = eig_vecs

        # DEBUG START
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # principal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        # principal_frame.transform(T_R_align)
        # T_R_test = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])
        # given_pos_trans_pc = copy.deepcopy(given_pos_pc)
        # given_pos_trans_pc.paint_uniform_color(np.array([[1],[1],[0]]))
        # given_pos_trans_pc.transform(T_R_align @ T_R_test @ np.linalg.inv(T_R_align))
        # vis.add_geometry(principal_frame)
        # vis.add_geometry(given_pos_pc)
        # vis.add_geometry(given_pos_trans_pc)
        # vis.run()
        # DEBUG END

        for R_90 in all_90_rotations():
            T_90 = np.eye(4)
            T_90[0:3, 0:3] = R_90

            T_init = T_R_align @ T_90 @ np.linalg.inv(T_R_align) @ T_ICP.copy()

            reg_result = o3d.pipelines.registration.registration_icp(
                given_pos_pc,
                desired_pos_pc,
                max_correspondence_distance=args.icp_max_corr_dist,
                criteria=icp_criteria,
                init=T_init,
            )

            rmse_list = np.append(rmse_list, reg_result.inlier_rmse)
            fitness_list = np.append(fitness_list, reg_result.fitness)
            # T_est = reg_result.transformation @ T_init
            T_est = reg_result.transformation
            T_list = np.append(T_list, np.reshape(T_est, (4, 4, 1)), axis=2)

        best_i = np.argmin(rmse_list)
        T_ICP = T_list[:, :, best_i]

    print("  - ICP:")
    print("    - fitness = {}".format(fitness_list[best_i]))
    print("    - inlier_RMSE = {}".format(rmse_list[best_i]))

    T_s = np.diag(np.array([s, s, s, 1.0]))

    T_t = np.eye(4)
    T_t[0:3, 3] = ct

    print("  - DEBUG: T_ICP")
    print(T_ICP)
    print("  - DEBUG: T_t")
    print(T_t)
    print("  - DEBUG: T_s")
    print(T_s)

    T = T_ICP @ T_t @ T_s
    # T = T_ICP @ np.linalg.inv(T_t) @ T_s
    # T = np.linalg.inv(T_ICP) @ np.linalg.inv(T_t) @ T_s

    print("  - transformation from the given to desired coordinate frame = ")
    np.set_printoptions(
        threshold=np.inf, precision=6, suppress=True, floatmode="fixed", linewidth=80
    )
    print(T)

    if args.vis_pos_align:
        print("- running Open3D visualization")
        given_pos_pc_orig = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(given_pos_arr_orig.T)
        )
        given_pos_pc_orig.paint_uniform_color(np.array([[0], [0], [1]]))
        given_pos_arr_trans = p2e(T @ e2p(given_pos_arr_orig))
        given_pos_pc_trans = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(given_pos_arr_trans.T)
        )
        given_pos_pc_trans.paint_uniform_color(np.array([[0], [1], [0]]))
        desired_pos_pc_orig = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(desired_pos_arr_orig.T)
        )
        desired_pos_pc_orig.paint_uniform_color(np.array([[1], [0], [0]]))

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(given_pos_pc_orig)
        vis.add_geometry(desired_pos_pc_orig)
        vis.add_geometry(given_pos_pc_trans)
        vis.run()

    print("- transforming the given COLMAP model")
    model_trans = copy.deepcopy(model_given)
    for img in model_trans.images.values():
        T_orig = np.eye(4)
        T_orig[0:3, 3] = img.tvec
        T_orig[0:3, 0:3] = img.rotmat()

        T_new = (
            T_s
            @ T_orig
            @ (np.linalg.inv(T_s) @ np.linalg.inv(T_t))
            @ np.linalg.inv(T_ICP)
        )
        img.tvec = T_new[0:3, 3]
        img.qvec = rotmat2quat(T_new[0:3, 0:3])

    print("- writing the transformed COLMAP model")
    model_trans.write(args.transformed_colmap_model)


def colmap2cam_pc(model):
    cam_pc = np.empty((3, 0))

    for img in model.images.values():
        R = img.rotmat()
        t = np.reshape(img.tvec, (3, 1))
        c = -R.T @ t

        cam_pc = np.append(cam_pc, c, axis=1)

    return cam_pc


def rotmat2quat(R):
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


def max_dist(pos_set):
    n = pos_set.shape[1]
    set_tile_c = np.tile(np.reshape(pos_set.T, (-1, 1, 3)), (1, n, 1))
    set_tile_r = np.tile(np.reshape(pos_set.T, (1, -1, 3)), (n, 1, 1))
    dist_arr = np.squeeze(np.sqrt(np.sum((set_tile_c - set_tile_r) ** 2, axis=2)))
    return np.amax(dist_arr)


def e2p(u_e):
    u_e = np.reshape(u_e, (u_e.shape[0], -1))
    u_p = np.concatenate((u_e, np.ones((1, u_e.shape[1]))), axis=0)
    return u_p


def p2e(u_p):
    u_p = np.reshape(u_p, (u_p.shape[0], -1))
    u_e = u_p[0:-1, :] / abs(u_p[-1, :])
    return u_e


def skew_sym(vec):
    vec = np.reshape(vec, (3, 1))
    return np.array(
        [
            [0, -vec[2, 0], vec[1, 0]],
            [vec[2, 0], 0, -vec[0, 0]],
            [-vec[1, 0], vec[0, 0], 0],
        ]
    )


def axis_ang2R(axis, ang):
    ang = math.radians(ang)
    axis = axis / np.sqrt(np.sum(axis**2))

    skew_sym_axis = skew_sym(axis)

    # - Rodrigues' rotation formula
    # R = I + sin(ang)*[v] + (1-cos(ang))*[v]^2
    return (
        np.eye(3)
        + math.sin(ang) * skew_sym_axis
        + (1 - math.cos(ang)) * (skew_sym_axis @ skew_sym_axis)
    )


# - code adapted from https://stackoverflow.com/a/70413438/10351620
def all_90_rotations():
    import itertools

    for x, y, z in itertools.permutations([0, 1, 2]):
        for sx, sy, sz in itertools.product([-1, 1], repeat=3):
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0, x] = sx
            rotation_matrix[1, y] = sy
            rotation_matrix[2, z] = sz
            if np.linalg.det(rotation_matrix) == 1:
                yield rotation_matrix


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
