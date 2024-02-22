#!/usr/bin/env python3


"""
Quickly visualize multiple 3D meshes in Open3D, allowing easy visual comparison.

"""


import argparse

import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser(description="3D mesh visualizer")
parser.add_argument("mesh_files", type=str, nargs="+", help="Paths to the input 3D mesh files")
parser.add_argument("--background_color", type=float, default=[255/255, 255/255, 255/255], nargs="+", 
    help="Background color of the visualization - default: %(default)s")
parser.add_argument("--show_back_face", action="store_true", help="Show the back face of the mesh")


color_palette = np.array([[1.0, 0.5, 0.5], [0.5, 1.0, 0.5], [0.5, 0.5, 1.0], [1.0, 0.75, 0.5], [0.5, 0.75, 1.0]])


def main(args):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    for idx, mesh_file in enumerate(args.mesh_files):
        print("Loading {}".format(mesh_file))
        mesh = o3d.io.read_triangle_mesh(mesh_file, print_progress=True)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color_palette[idx % len(color_palette), :])
        vis.add_geometry(mesh)

    vis.get_render_option().background_color = np.array(args.background_color)
    vis.get_render_option().mesh_show_back_face = args.show_back_face
    vis.run()

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)