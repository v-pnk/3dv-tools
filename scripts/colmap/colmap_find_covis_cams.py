#!/usr/bin/env python3


"""
Find cameras (images in COLMAP) which have common 3D points with a given camera
and export a new COLMAP model with only these cameras.
"""


import os
import copy
import argparse
from tqdm import tqdm
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument("input_colmap", type=str,
                    help="Input COLMAP model directory")
parser.add_argument("output_colmap", type=str,
                    help="Output COLMAP model directory")
parser.add_argument("--img_name", type=str,
                    help="\"Source\" image name to find covisibility for (using first image in the model if not specified)")
parser.add_argument("--do_not_include_src", type=str,
                    help="Do not include the source image in the output model")
parser.add_argument("--min_shared_pnts", type=int, default=5,
                    help="Minimum number of shared 3D points between two images to be considered covisible")


def main(args):
    assert os.path.isdir(args.input_colmap), "Input COLMAP model directory \"{}\" does not exist".format(args.input_colmap)
    assert os.path.isdir(args.output_colmap), "Output COLMAP model directory \"{}\" does not exist".format(args.output_colmap)

    print("- reading COLMAP model")
    model = pycolmap.Reconstruction(args.input_colmap)

    if args.img_name is None:
        img_src = list(model.images.values())[0]
    else:
        img_src = model.find_image_with_name(args.img_name)
        assert img_src is not None, "Given image name \"{}\" not found in COLMAP model".format(args.img_name)
    
    print("- image {} selected as the source image".format(img_src.name))

    # get all 3D points visible from the camera
    print("- extracting 3D points of the source image")
    points2D_src = img_src.points2D
    points3D_src_ids = [p2d.point3D_id for p2d in points2D_src if p2d.has_point3D()]
    points3D_src = [model.points3D[p3d_id] for p3d_id in points3D_src_ids]

    # get all images which have common 3D points with the camera
    print("- searching for covisible images")
    covisible_imgs = set()

    if args.do_not_include_src is None:
        covisible_imgs.add(img_src.image_id)

    print("  - filtering based on individual 3D point visibility")
    for point3D in tqdm(points3D_src):
        track_image_ids = [trck_elem.image_id for trck_elem in point3D.track.elements if trck_elem.image_id != img_src.image_id]
        covisible_imgs |= set(track_image_ids)
    
    # get rid of images which have less than given number of covisible 3D points
    if args.min_shared_pnts > 1:
        print("  - filtering based on number of covisible 3D points")
        covisible_imgs_new = set()
        for img_id in tqdm(covisible_imgs):
            img = model.images[img_id]
            points3D_ids = [p2d.point3D_id for p2d in img.points2D if p2d.has_point3D()]
            shared_pnts = set(points3D_ids) & set(points3D_src_ids)
            if len(shared_pnts) >= args.min_shared_pnts:
                covisible_imgs_new |= set([img_id])
        
        covisible_imgs = covisible_imgs_new
    
    print("- generating new COLMAP model")
    new_model = copy.deepcopy(model)
    for img_src_id in model.images.keys():
        if img_src_id not in covisible_imgs:
            new_model.deregister_image(img_src_id)

    print("- writing new COLMAP model")
    new_model.write_binary(args.output_colmap)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)