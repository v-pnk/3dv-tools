#!/usr/bin/env python3


"""
Subsample the given point cloud using MeshLab
"""


import os
import argparse
import pymeshlab


parser = argparse.ArgumentParser(description="Evaluation tool")
parser.add_argument("input_pc", type=str,
    help="Path to the input point cloud")
parser.add_argument("output_pc", type=str,
    help="Path to the output point cloud")
parser.add_argument("frac", type=float,
    help="Fraction of the points to keep")


def main(args):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(args.input_pc)
    curr_mesh = ms.current_mesh()
    samplenum = int(args.frac * curr_mesh.vertex_number())
    ms.generate_simplified_point_cloud(samplenum=samplenum)
    ms.save_current_mesh(args.output_pc)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
