#!/usr/bin/env python3


"""
Simple script for computation of centroid and bounding box of vertices of 
the given 3D model. The model can be a point cloud or a mesh.

"""


import os
import argparse
import numpy as np
import open3d as o3d


parser = argparse.ArgumentParser(description="")
parser.add_argument("input_file", type=str, help="Path to the input 3D file parsable by Open3D")


mesh_exts = ('.stl', '.obj', '.off', '.gltf')


def main(args):
    assert os.path.exists(args.input_file), "The given file does not exist:{}".format(args.input_file)

    ext = os.path.splitext(args.input_file)[1]
    if ext in mesh_exts:
        # load the input file as mesh
        mesh = o3d.io.read_triangle_mesh(args.input_file, print_progress=True)
        vertices = np.asarray(mesh.vertices).T
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(vertices.T)
    else:
        # load the input file as point cloud
        pc = o3d.io.read_point_cloud(args.input_file, print_progress=True)

    centroid = pc.get_center()
    aabb = pc.get_axis_aligned_bounding_box()

    print("centroid:")
    print(centroid)
    print('axis-aligned bounding box corner points:')
    print(aabb.get_print_info())
    print('axis-aligned bounding box extent:')
    print(aabb.get_extent())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
