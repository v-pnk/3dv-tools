#!/usr/bin/python3


"""
Sample camera poses for rendering of a 3D model. The poses are sampled in the
axis-aligned bounding box of the model and are conditioned by the visibility
of the model from the camera position and orientation.

"""


import os
import copy
import math
import numpy as np
import open3d as o3d
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("mesh_path", type=str, help="Path to the mesh file")

parser.add_argument("--vrephoto_dir", type=str,help="Path to the output directory for .cam and .res files")
parser.add_argument("--out_colmap_model", type=str, help="Path to the directory where will be created the COLMAP model")

parser.add_argument("--up_vec", type=float, nargs="+", default=[0,0,1], help="Up vector (Z+) in the mesh coordinate frame (vector of length 3)")

parser.add_argument("--pos_sampling_num", type=int, default=1200, help="Number of initial camera position samples (in the mesh axis-aligned bounding box)")
parser.add_argument("--pos_sampling_mode", type=str, default="random", choices=["regular", "random"], help="Camera positions sampling method (regular/random).")
parser.add_argument("--bbox_margin_limits", type=float, nargs="+", default=[0,0,0,0,0,0], help="Margin from the mesh bounding box limits (X-,Y-,Z-,X+,Y+,Z+)")

parser.add_argument("--ang_sampling_num", type=int, default=3, help="Number of initial orientaion samples per single camera position sample")
parser.add_argument("--ang_sampling_mode", type=str, default="regular_rand_yaw_pitch", choices=["regular", "regular_rand_yaw", "regular_rand_yaw_pitch", "random_yaw", "random_yaw_pitch"], help="Camera orientation sampling method.")
parser.add_argument("--ang_yaw_std", type=float, default=15, help="Yaw standard deviation (if using random yaw) [degrees]")
parser.add_argument("--ang_pitch_std", type=float, default=15, help="Pitch standard deviation (if using random pitch) [degrees]")

# - camera position validity conditions
parser.add_argument("--zm_limits", type=float, nargs="+", default=[0.6, 5], help="Distance of the camera center from the floor (-Z)")
parser.add_argument("--zp_limits", type=float, nargs="+", default=[0.6, 5], help="Distance of the camera center from the ceiling (+Z)")
parser.add_argument("--xy_limits", type=float, nargs="+", default=[0.2, 1000.0], help="Distance of the camera center from the walls (X,Y)")
parser.add_argument("--xy_valid_samples", type=int, default=4, help="Number of horizontal samples which have to be within the xy_limits from walls")
parser.add_argument("--check_normals", type=bool, default=True, help="Check if the casted ray intersects the mesh surface from the front or from the back side")

# - camera orientation validity conditions
parser.add_argument("--d_limits", type=float, nargs="+", default=[2.5, 1000], help="Limits on the length of the camera principal ray")

# - fixed camera intrinsics values
parser.add_argument("--img_width", type=int, default=800, help="Image width - default: %(default)s")
parser.add_argument("--img_height", type=int, default=800, help="Image height - default: %(default)s")
parser.add_argument("--cam_foclen", type=float, default=800, help="Image height - default: %(default)s")

parser.add_argument("--dont_visualize", action='store_true', help="Do not visualize the generated camera poses.")


EPS = 1e-12


def main(args):
    assert len(args.up_vec) == 3, "ERROR: The direction vector needs 3 elements."
    args.up_vec = normalize(np.array(args.up_vec))

    print("- reading the mesh")
    mesh = o3d.io.read_triangle_mesh(args.mesh_path)

    # - rotate the mesh to match the up vector
    (axis_up, ang_up) = vect_vect2axis_ang(np.array([0,0,1]), args.up_vec)
    R_up = axis_ang2R(axis_up, ang_up)
    mesh.rotate(R_up, mesh.get_center())
    
    mesh_min_bounds = mesh.get_min_bound() + np.array(args.bbox_margin_limits[0:3])
    mesh_max_bounds = mesh.get_max_bound() - np.array(args.bbox_margin_limits[3:6])
    
    # - generate the initial camera positions
    print("- generating the initial camera positions")
    if args.pos_sampling_mode == "regular":
        # - sample in regular grid
        mesh_aabb_volume = np.prod(mesh_max_bounds - mesh_min_bounds)
        mesh_aabb_size = mesh_max_bounds - mesh_min_bounds
        mesh_center = 0.5*(mesh_max_bounds + mesh_min_bounds)

        sample_spacing = (mesh_aabb_volume / args.pos_sampling_num) ** (1.0 / 3.0)
        sample_num_axes = (mesh_aabb_size / sample_spacing).astype(np.int32) + 2

        samples_x = mesh_center[0] + np.linspace(-0.5*(sample_num_axes[0]-1)*sample_spacing, 0.5*(sample_num_axes[0]-1)*sample_spacing, sample_num_axes[0])
        samples_y = mesh_center[1] + np.linspace(-0.5*(sample_num_axes[1]-1)*sample_spacing, 0.5*(sample_num_axes[1]-1)*sample_spacing, sample_num_axes[1])
        samples_z = mesh_center[2] + np.linspace(-0.5*(sample_num_axes[2]-1)*sample_spacing, 0.5*(sample_num_axes[2]-1)*sample_spacing, sample_num_axes[2])
        sample_grid = np.meshgrid(samples_x, samples_y, samples_z)
        
        pos_samples = np.reshape(sample_grid, (3,-1))

    elif args.pos_sampling_mode == "random":
        # - sample randomly
        samples_x = np.random.uniform(low=mesh_min_bounds[0], high=mesh_max_bounds[0], size=(args.pos_sampling_num))
        samples_y = np.random.uniform(low=mesh_min_bounds[1], high=mesh_max_bounds[1], size=(args.pos_sampling_num))
        samples_z = np.random.uniform(low=mesh_min_bounds[2], high=mesh_max_bounds[2], size=(args.pos_sampling_num))
        pos_samples = np.vstack((samples_x, samples_y, samples_z))

    init_cam_positions = copy.deepcopy(pos_samples)
    
    print("- generated {:d} initial camera positions".format(pos_samples.shape[1]))

    print("- casting rays around the initial camera positions")
    # -X, -Y, -Z, +X, +Y, +Z
    ray_directions = np.hstack((-np.eye(3),np.eye(3)))
    zm_idx = np.tile([False, False, True, False, False, False], (pos_samples.shape[1]))
    zp_idx = np.tile([False, False, False, False, False, True], (pos_samples.shape[1]))
    xy_idx = np.tile([True, True, False, True, True, False], (pos_samples.shape[1]))

    # - cast rays from each camera position and check which are valid
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    ray_centers = np.repeat(pos_samples, ray_directions.shape[1], axis=1)
    ray_directions = np.tile(ray_directions, (1, pos_samples.shape[1]))
    rays = o3d.core.Tensor(np.vstack((ray_centers, ray_directions)).T, dtype=o3d.core.Dtype.Float32)

    rc_dict = scene.cast_rays(rays)

    print("- filtering the initial camera positions")
    ray_lengths = rc_dict['t_hit'].numpy()
    hit_normals = rc_dict["primitive_normals"].numpy().T

    zm_ray_lenghts = ray_lengths[zm_idx]
    zp_ray_lenghts = ray_lengths[zp_idx]
    xy_ray_lenghts = ray_lengths[xy_idx]
    
    zm_valid = np.logical_and(zm_ray_lenghts >= args.zm_limits[0], zm_ray_lenghts <= args.zm_limits[1])
    zp_valid_min = zp_ray_lenghts >= args.zp_limits[0]
    
    if args.zp_limits[1] > 0:
        zp_valid_max = zp_ray_lenghts <= args.zp_limits[1]
    else:
        zp_valid_max = np.full_like(zp_valid_min, True)
    zp_valid = np.logical_and(zp_valid_min, zp_valid_max)
    xy_valid = np.logical_and(xy_ray_lenghts >= args.xy_limits[0], xy_ray_lenghts <= args.xy_limits[1])

    if args.check_normals:
        # - the normal of the mesh and the ray direction should point into opposite directions
        #   - if the ray did not hit anything, the normal is zero vector
        normals_valid = np.einsum('ij,ij->j', ray_directions, hit_normals) <= 0.0
        zm_valid = np.logical_and(zm_valid, normals_valid[zm_idx])
        zp_valid = np.logical_and(zp_valid, normals_valid[zp_idx])
        xy_valid = np.logical_and(xy_valid, normals_valid[xy_idx])

    xy_valid = np.reshape(xy_valid, (-1, 4))
    xy_valid = np.sum(xy_valid, axis=1) >= args.xy_valid_samples

    valid_samples = np.logical_and(np.logical_and(zm_valid, zp_valid), xy_valid)
    pos_samples = pos_samples[:, valid_samples]

    valid_cam_positions = copy.deepcopy(pos_samples)

    print("- generated {:d} valid camera positions".format(pos_samples.shape[1]))

    # - generate the initial camera orientations
    print("- generating the initial camera orientations")
    cams_T = np.empty((4,4,0), dtype=np.float32)
    ray_directions = np.empty((3,0), dtype=np.float32)

    if args.ang_sampling_mode in ["regular", "regular_rand_yaw", "regular_rand_yaw_pitch"]:
        # - sample regularly in yaw
        yaw_samples = np.linspace(0, 360, args.ang_sampling_num, endpoint=False)
        yaw_samples = np.tile(yaw_samples, (pos_samples.shape[1]))
        pitch_samples = np.full_like(yaw_samples, 90)

        if args.ang_sampling_mode in ["regular_rand_yaw", "regular_rand_yaw_pitch"]:
            yaw_samples = yaw_samples + np.random.normal(loc=0.0, scale=args.ang_yaw_std, size=(yaw_samples.shape[0]))

            if args.ang_sampling_mode in ["regular_rand_yaw_pitch"]:
                pitch_samples = pitch_samples + np.random.normal(loc=0.0, scale=args.ang_pitch_std, size=(pitch_samples.shape[0]))

    elif args.ang_sampling_mode in ["random_yaw", "random_yaw_pitch"]:
        yaw_samples = np.random.uniform(low=0, high=360, size=(args.ang_sampling_num*pos_samples.shape[1]))
        pitch_samples = np.full_like(yaw_samples, -90)

        if args.ang_sampling_mode in ["random_yaw_pitch"]:
            pitch_samples = pitch_samples + np.random.normal(loc=0.0, scale=args.ang_pitch_std, size=(pitch_samples.shape[0]))
    
    pos_samples = np.repeat(pos_samples, args.ang_sampling_num, axis=1)

    yaw_samples = np.radians(yaw_samples)
    pitch_samples = np.radians(pitch_samples)
    
    yc = np.cos(yaw_samples)
    ys = np.sin(yaw_samples)
    pc = np.cos(pitch_samples)
    ps = np.sin(pitch_samples)
    
    R_yaw = np.eye(3)
    R_yaw = np.tile(R_yaw, (yaw_samples.shape[0], 1, 1))
    R_yaw[:,0,0] = yc
    R_yaw[:,0,1] = -ys
    R_yaw[:,1,0] = ys
    R_yaw[:,1,1] = yc

    R_pitch = np.eye(3)
    R_pitch = np.tile(R_pitch, (pitch_samples.shape[0], 1, 1))
    R_pitch[:,1,1] = pc
    R_pitch[:,1,2] = -ps
    R_pitch[:,2,1] = ps
    R_pitch[:,2,2] = pc

    R = R_up.T @ R_pitch @ R_yaw
    t = np.swapaxes(np.expand_dims(pos_samples, axis=0), 0, 2)
    R = np.swapaxes(R, 0, 2)
    t = np.swapaxes(t, 0, 2)

    cams_T = np.eye(4)
    cams_T = np.expand_dims(cams_T, 2)
    cams_T = np.tile(cams_T, (1, 1, R.shape[2]))
    cams_T[0:3, 0:3, :] = R
    cams_T[0:3, 3, :] = t

    # TODO: test multiple rays per view to find views looking into empty space
    ray_directions = np.swapaxes(np.squeeze(np.linalg.inv(R_yaw) @ np.linalg.inv(R_pitch) @ np.array([[0, 0, 1]]).T), 0, 1)

    print("- casting the principal camera rays")
    rays = o3d.core.Tensor(np.vstack((pos_samples, ray_directions)).T, dtype=o3d.core.Dtype.Float32)

    rc_dict = scene.cast_rays(rays)

    print("- filtering the initial camera orientations")
    ray_lengths = rc_dict['t_hit'].numpy()

    d_valid_min = ray_lengths >= args.d_limits[0]
    if args.d_limits[1] > 0:
        d_valid_max = ray_lengths <= args.d_limits[1]
    else:
        d_valid_max = np.full_like(d_valid_min, True)
    d_valid = np.logical_and(d_valid_min, d_valid_max)

    cams_T = cams_T[:,:,d_valid]

    print("- generated {:d} valid camera poses".format(cams_T.shape[2]))

    # - generate the output vrephoto camera definitions
    if args.vrephoto_dir is not None:
        print("- exporting the camera definitions in vrephoto format")
        write_vrephoto(args.vrephoto_dir, cams_T, args.cam_foclen, args.img_width, args.img_height)

    # - generate the output COLMAP model
    if args.out_colmap_model is not None:
        print("- exporting the camera definitions in COLMAP format")
        write_colmap(args.out_colmap_model, cams_T, args.cam_foclen, args.img_width, args.img_height)

    if not(args.dont_visualize):
        print("- visualizing")
        # Visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        # - world CS
        o3d_world_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

        # - principal axis rays
        finite_rays = np.isfinite(ray_lengths)
        ray_lengths = ray_lengths[finite_rays]
        ray_directions = ray_directions[:,finite_rays]
        pos_samples = pos_samples[:,finite_rays]

        o3d_ray_corrs = [(i,i) for i in range(pos_samples.shape[1])]
        o3d_origins = o3d.geometry.PointCloud()
        o3d_origins.points = o3d.utility.Vector3dVector(pos_samples.T)
        o3d_hits = o3d.geometry.PointCloud()
        o3d_hits.points = o3d.utility.Vector3dVector((pos_samples+(ray_directions*ray_lengths)).T)

        o3d_rays = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            o3d_origins, o3d_hits, o3d_ray_corrs)
        o3d_rays.paint_uniform_color([0.0, 0.0, 0.6])
        vis.add_geometry(o3d_rays)

        # - valid camera poses
        for cam_i in np.arange(cams_T.shape[2]):
            o3d_cam_point = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            o3d_cam_point.paint_uniform_color([0.0, 0.4, 0.2])
            T = np.eye(4)
            T[0:3, 3] = cams_T[0:3, 3, cam_i]
            o3d_cam_point.transform(T)
            # vis.add_geometry(o3d_cam_point)

            o3d_cam_cs = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            T = cams_T[:,:,cam_i]
            o3d_cam_cs.transform(T)
            vis.add_geometry(o3d_cam_cs)

            K = np.array([[args.cam_foclen, 0, args.img_width/2], [0, args.cam_foclen,args.img_height/2], [0, 0, 1]])
            T = np.linalg.inv(cams_T[:, :, cam_i])
            o3d_cam_vis = o3d.geometry.LineSet.create_camera_visualization(args.img_width, args.img_height, K, T, scale=0.1)
            vis.add_geometry(o3d_cam_vis)

        # - model mesh
        o3d_mesh = copy.deepcopy(mesh)

        mesh_verts_z = np.asarray(o3d_mesh.vertices).T[2,:]
        color_vals = (mesh_verts_z - mesh_verts_z.min()) / (mesh_verts_z.max() - mesh_verts_z.min())
        colors = np.tile(color_vals, (3,1))
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors.T)
        o3d_mesh.rotate(R_up.T, o3d_mesh.get_center())

        vis.add_geometry(o3d_world_cs)
        vis.add_geometry(o3d_mesh)
        
        vis.run()
    

def vect_vect2axis_ang(vec_1, vec_2):
    vec_1 = np.ascontiguousarray(vec_1)
    vec_2 = np.ascontiguousarray(vec_2)
    vec_1 = normalize(np.reshape(vec_1, (3, 1)))
    vec_2 = normalize(np.reshape(vec_2, (3, 1)))

    # - if the vectors are same, cross product is zero vector and its norm
    #   produces NaN values
    # if np.isclose(vec_1, vec_2).all():
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
        axis = np.reshape(
            np.cross(vec_1.flatten(), rand_vect.flatten()), (3, 1))
        # - angle is zero
        ang = 0.0
    else:
        cross_prod = np.cross(vec_1.flatten(), vec_2.flatten())
        dot_prod = np.dot(vec_1.flatten(), vec_2.flatten())
        axis = normalize(np.reshape(cross_prod, (3, 1)))

        # cos(ang) = (u . v) / (|u| * |v|) = (u . v) / (1 * 1) = u . v
        # ang_cos = dot_prod / (vect_len(vec_1)*vect_len(vec_2))
        ang_cos = dot_prod
        # sin(ang) = |u x v| / (|u| * |v|) = |u x v| / (1 * 1) = |u x v|
        # ang_sin = vect_len(cross_prod) / (vect_len(vec_1)*vect_len(vec_2))
        ang_sin = vect_len(cross_prod)
        # use atan2 to compute the angle
        ang = math.degrees(math.atan2(ang_sin, ang_cos))

    return (axis, ang)


def axis_ang2R(axis, ang):
    ang = math.radians(ang)
    axis = normalize(axis)

    skew_sym_axis = skew_sym(axis)

    # - Rodrigues' rotation formula
    # R = I + sin(ang)*[v] + (1-cos(ang))*[v]^2
    return np.eye(3) + math.sin(ang)*skew_sym_axis + (1-math.cos(ang))*(skew_sym_axis @ skew_sym_axis)


def vect_len(vect):
    return np.sqrt(vect_len_sq(vect))


def vect_len_sq(vect):
    return np.sum(vect*vect, axis=0)


def normalize(vect):
    if vect.ndim == 1:
        return vect / (np.sqrt(np.sum(vect*vect)) + EPS)
    elif vect.ndim == 2:
        return vect / (np.reshape(np.sqrt(np.sum(vect*vect, axis=0)), (1, -1)) + EPS)


def skew_sym(vec):
    vec = np.reshape(vec, (3, 1))
    return np.array([[0, -vec[2, 0], vec[1, 0]], [vec[2, 0], 0, -vec[0, 0]], [-vec[1, 0], vec[0, 0], 0]])


def R_center2T(R, center):
    center = np.reshape(center, (3,1))

    Ta = np.eye(4)
    Ta[0:3, 3] = -center.flatten()
    
    Tb = np.eye(4)
    Tb[0:3, 0:3] = R

    Tc = np.eye(4)
    Tc[0:3, 3] = center.flatten()

    return (Tc @ Tb @ Ta)


def p2e(u_p):
    u_p = np.reshape(u_p, (u_p.shape[0], -1))
    u_e = u_p[0:-1, :] / abs(u_p[-1, :])
    return u_e


def e2p(u_e):
    u_e = np.reshape(u_e, (u_e.shape[0], -1))
    u_p = np.concatenate((u_e, np.ones((1, u_e.shape[1]))), axis=0)
    return u_p


def write_vrephoto(vrephoto_dir, T_cams, f, w, h):
    for cam_i in np.arange(T_cams.shape[2]):
        image_name = "surf_cam_{:0>6d}".format(cam_i)
        T = T_cams[:,:,cam_i]

        # from world2cam to cam2world
        T = np.linalg.inv(T)

        cam_file_path = os.path.join(vrephoto_dir, "{}.cam".format(image_name))
        res_file_path = os.path.join(vrephoto_dir, "{}.res".format(image_name))
        
        f_cam = open(cam_file_path, 'w')
        # - cam file format:
        #   t0 t1 t2 R00 R01 R02 R10 R11 R12 R20 R21 R22
        #   f_norm 0 0 f_aspect cx/w (h-cy)/h
        pose_data = [T[0, 3], T[1, 3], T[2, 3],
                     T[0, 0], T[0, 1], T[0, 2],
                     T[1, 0], T[1, 1], T[1, 2],
                     T[2, 0], T[2, 1], T[2, 2]]
        cam_line_1st = " ".join(map(str, pose_data))

        cx = 0.5 * w
        cy = 0.5 * h
        f_norm = f / float(max(w, h))
        f_aspect = 1.0
        cam_file_data = [f_norm, 0, 0, f_aspect, cx / w, (h - cy) / h]
        cam_line_2nd = " ".join(map(str, cam_file_data))

        f_cam.write(cam_line_1st + "\n")
        f_cam.write(cam_line_2nd)
        f_cam.close()

        f_res = open(res_file_path, 'w')
        # - res file format:
        #   w h
        res_line = " ".join(map(str, [w, h]))
        f_res.write(res_line + "\n")
        f_res.close()
    

def write_colmap(colmap_dir, T_cams, f, w, h):
    colmap_images_path = os.path.join(colmap_dir, "images.txt")
    colmap_cameras_path = os.path.join(colmap_dir, "cameras.txt")
    colmap_points3D_path = os.path.join(colmap_dir, "points3D.txt")

    f_points3D = open(colmap_points3D_path, 'w')
    f_points3D.close()

    f_cameras = open(colmap_cameras_path, 'w')
    f_cameras.write("# Camera list with one line of data per camera:\n")
    f_cameras.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f_cameras.write("# Number of cameras: 1\n")
    f_cameras.write("1 PINHOLE {} {} {} {} {} {}".format(w, h, f, f, 0.5*w, 0.5*h))
    f_cameras.close()

    f_images = open(colmap_images_path, 'w')
    f_images.write("# Image list with two lines of data per image:\n")
    f_images.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f_images.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f_images.write("# Number of images: {}, mean observations per image: 0.0\n".format(T_cams.shape[2]))

    for img_i in np.arange(T_cams.shape[2]):
        T = np.linalg.inv(T_cams[:,:,img_i])
        qvec = R2quat(T[0:3, 0:3]).flatten()
        tvec = T[0:3, 3]
        img_name = "surf_cam_{:0>6d}_rendered_color.png".format(img_i)
        f_images.write("{} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f} 1 {}\n\n".format(
            img_i, qvec[0], qvec[1], qvec[2], qvec[3],
            tvec[0], tvec[1], tvec[2], img_name))
    
    f_images.close()


def R2quat(R):
    tr = np.trace(R)

    if (tr > 0):
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif ((R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif (R[1, 1] > R[2, 2]):
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
