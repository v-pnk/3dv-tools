#!/usr/bin/env python3


"""
Quickly visualize multiple 3D meshes in Open3D, allowing easy visual comparison.
The mesh colors are printed in the terminal.

"""


import argparse

import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser(description="3D mesh visualizer")
parser.add_argument(
    "mesh_files",
    type=str,
    nargs="+",
    help="Paths to the input 3D mesh or point cloud files",
)
parser.add_argument(
    "--background_color",
    type=float,
    default=[255 / 255, 255 / 255, 255 / 255],
    nargs="+",
    help="Background color of the visualization - default: %(default)s",
)
parser.add_argument(
    "--show_back_face", 
    action="store_true", 
    help="Show the back face of the mesh",
)
parser.add_argument(
    "--output_mesh",
    type=str,
    default=None,
    help="Path to the output mesh file",
)


color_palette = np.array(
    [
        [1.0, 0.5, 0.5],
        [0.5, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [1.0, 0.75, 0.5],
        [0.5, 0.75, 1.0],
    ]
)


def main(args):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    color_print_str = ""

    input_types = []
    input_data = []

    for idx, mesh_file in enumerate(args.mesh_files):
        print("Loading {}".format(mesh_file))
        mesh = o3d.io.read_triangle_mesh(mesh_file, print_progress=True)

        # - check if it's a mesh or a point cloud
        if len(mesh.triangles) > 0:
            mesh.compute_vertex_normals()
            input_types.append("mesh")
        else:
            mesh_tmp = o3d.geometry.PointCloud()
            mesh_tmp.points = o3d.utility.Vector3dVector(mesh.vertices)
            mesh = mesh_tmp
            input_types.append("pc")

        color = color_palette[idx % len(color_palette), :]
        mesh.paint_uniform_color(color)

        vis.add_geometry(mesh)
        input_data.append(mesh)

        color_print_str += (
            "\x1b[38;2;{};{};{}m".format(
                int(255 * color[0]), int(255 * color[1]), int(255 * color[2])
            )
            + 4 * "\u2588"
            + "\033[0m"
        )
        color_print_str += "  " + mesh_file + "\n"

    print(color_print_str)

    if args.output_mesh is not None:
        if len(set(input_types)) == 1 and set(input_types) == {"pc"}:
            output_pc = o3d.geometry.PointCloud()
            for d in input_data:
                output_pc += d
            print("Saving the output point cloud to {}".format(args.output_mesh))
            o3d.io.write_point_cloud(args.output_mesh, output_pc)
        elif len(set(input_types)) == 1 and set(input_types) == {"mesh"}:
            output_mesh = o3d.geometry.TriangleMesh()
            for d in input_data:
                output_mesh += d
            print("Saving the output mesh to {}".format(args.output_mesh))
            o3d.io.write_triangle_mesh(args.output_mesh, output_mesh)
        else:
            output_mesh = o3d.geometry.TriangleMesh()
            for d in input_data:
                if type(d) == o3d.geometry.PointCloud:
                    m = o3d.geometry.TriangleMesh()
                    m.vertices = d.points
                    m.vertex_colors = d.colors
                    output_mesh += m
                else:
                    output_mesh += d
            
            print("Saving the output mesh to {}".format(args.output_mesh))
            print("WARN: The output file is a combination of the given meshes and point clouds. Visualization tools will probably not interpret the output file correctly.")
            o3d.io.write_triangle_mesh(args.output_mesh, output_mesh)

    vis.get_render_option().background_color = np.array(args.background_color)
    vis.get_render_option().mesh_show_back_face = args.show_back_face
    vis.run()


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
