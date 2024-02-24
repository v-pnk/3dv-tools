#!/usr/bin/env python3


"""
Compute distance between two 3D meshes.

"""


import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import faiss


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "mesh_A", 
    type=str, 
    help="Path to the first mesh (visualized)"
)
parser.add_argument(
    "mesh_B", 
    type=str, 
    help="Path to the second mesh (reference)"
)
parser.add_argument(
    "--auto_mesh_A_tris", 
    type=int, 
    help="Desired number of triangles of mesh A"
)
parser.add_argument(
    "--auto_mesh_B_tris", 
    type=int, 
    help="Desired number of triangles of mesh A"
)
parser.add_argument(
    "--simplify_frac_A",
    type=float,
    help="Fraction of vertices to keep after simplification step on mesh A",
)
parser.add_argument(
    "--simplify_frac_B",
    type=float,
    help="Fraction of vertices to keep after simplification step on mesh B",
)
parser.add_argument(
    "--subdiv_iter_A",
    type=int,
    help="Number of subdivision algorithm iterations on mesh A (single iteration divides each triangle to four)",
)
parser.add_argument(
    "--subdiv_iter_B",
    type=int,
    help="Number of subdivision algorithm iterations on mesh B (single iteration divides each triangle to four)",
)
parser.add_argument(
    "--max_dist", 
    default=1.0, 
    type=float, 
    help="Set maximum distance (-1 for auto)"
)
parser.add_argument(
    "--min_dist", 
    default=0.0, 
    type=float, 
    help="Set minimum distance (-1 for auto)"
)
parser.add_argument(
    "--dist_fnc",
    default="lin",
    choices=["lin", "root2", "root4", "root8"],
    type=str,
    help="Function to apply on the distances",
)
parser.add_argument(
    "--show_dist_hist", 
    action="store_true", 
    help="Show the histogram of distances"
)
parser.add_argument(
    "--dont_visualize", 
    action="store_false", 
    help="Do not show the mesh visualization"
)
parser.add_argument(
    "--background_color",
    type=float,
    default=[255 / 255, 255 / 255, 255 / 255],
    nargs="+",
    help="Background color of the visualization - default: %(default)s",
)
parser.add_argument(
    "--dist_path", 
    type=str, 
    help="Path for saving the computed distances in .npy file"
)


def main(args):
    mesh_A = o3d.io.read_triangle_mesh(args.mesh_A, print_progress=True)
    mesh_B = o3d.io.read_triangle_mesh(args.mesh_B, print_progress=True)

    tris_A_num_init = np.asarray(mesh_A.triangles).shape[0]
    tris_B_num_init = np.asarray(mesh_B.triangles).shape[0]

    print(tris_A_num_init)
    print(tris_B_num_init)

    if args.auto_mesh_A_tris is not None:
        if tris_A_num_init > args.auto_mesh_A_tris:
            mesh_A = mesh_A.simplify_quadric_decimation(args.auto_mesh_A_tris)
        elif tris_A_num_init < args.auto_mesh_A_tris:
            divide_iters = round(math.log(args.auto_mesh_A_tris / tris_A_num_init, 4))
            if divide_iters > 0:
                mesh_A = mesh_A.subdivide_midpoint(divide_iters)

    if args.auto_mesh_B_tris is not None:
        if tris_B_num_init > args.auto_mesh_B_tris:
            mesh_B = mesh_B.simplify_quadric_decimation(args.auto_mesh_B_tris)
        elif tris_B_num_init < args.auto_mesh_B_tris:
            divide_iters = round(math.log(args.auto_mesh_B_tris / tris_B_num_init, 4))
            if divide_iters > 0:
                mesh_B = mesh_B.subdivide_midpoint(divide_iters)

    if (args.auto_mesh_A_tris is None) and (args.subdiv_iter_A is not None):
        mesh_A = mesh_A.subdivide_midpoint(args.subdiv_iter_A)

    if (args.auto_mesh_B_tris is None) and (args.subdiv_iter_B is not None):
        mesh_B = mesh_B.subdivide_midpoint(args.subdiv_iter_B)

    tris_A_num = np.asarray(mesh_A.triangles).shape[0]
    tris_B_num = np.asarray(mesh_B.triangles).shape[0]

    if (args.auto_mesh_A_tris is None) and (args.simplify_frac_A is not None):
        mesh_A = mesh_A.simplify_quadric_decimation(
            int(args.simplify_frac_A * tris_A_num)
        )

    if (args.auto_mesh_B_tris is None) and (args.simplify_frac_B is not None):
        mesh_B = mesh_B.simplify_quadric_decimation(
            int(args.simplify_frac_B * tris_B_num)
        )

    tris_A_num = np.asarray(mesh_A.triangles).shape[0]
    tris_B_num = np.asarray(mesh_B.triangles).shape[0]
    verts_A = np.asarray(mesh_A.vertices).astype(np.float32)
    verts_B = np.asarray(mesh_B.vertices).astype(np.float32)

    print("- mesh A: {} --> {} triangles".format(tris_A_num_init, tris_A_num))
    print("- mesh B: {} --> {} triangles".format(tris_B_num_init, tris_B_num))

    # - use 3D L2 metric
    index = faiss.IndexFlatL2(3)
    # - add the vertices of mesh B to the index
    print("- creating FAISS index")
    index.add(verts_B)
    # - search for the nearest vertex from B to each vertex from A
    print("- computing the distances")
    dists, _ = index.search(verts_A, 1)

    # dists ~ [N x 1]
    dists.flatten()

    print("- min dist: {:.3f}".format(np.min(dists)))
    print("- max dist: {:.3f}".format(np.max(dists)))

    if args.show_dist_hist:
        plt.hist(100.0 * dists, bins=100, density=True, color=(0.002, 0.709, 0.923))
        if args.min_dist is not None:
            plt.xlim(left=args.min_dist)
        else:
            plt.xlim(left=0)
        if abs(args.max_dist + 1) > 10 * np.finfo(float).eps:
            plt.xlim(right=args.max_dist)
        plt.grid()
        plt.show()

    if args.dist_path is not None:
        if args.max_dist > 0:
            dists_out = dists[dists < args.max_dist]
        np.save(args.dist_path, dists_out)

    if args.max_dist > 0:
        dists = np.minimum(dists, args.max_dist)

    if args.dist_fnc.startswith("root"):
        for _ in range(int(np.log2(int(args.dist_fnc[4:])))):
            dists = np.sqrt(dists)

    dists_min = np.min(dists)
    dists_max = np.max(dists)
    dists_norm = (dists - dists_min) / (dists_max - dists_min)

    cmap = plt.get_cmap("viridis")
    rgb = cmap(dists_norm).squeeze()[:, 0:3]
    mesh_A.vertex_colors = o3d.utility.Vector3dVector(rgb)

    if args.dont_visualize:
        vis = o3d.visualization.Visualizer()

        vis.create_window()
        vis.add_geometry(mesh_A)

        vis.get_render_option().mesh_show_back_face = True
        vis.get_render_option().background_color = np.array(args.background_color)
        vis.run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
