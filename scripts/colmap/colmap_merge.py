#!/usr/bin/env python3

"""
Merges multiple COLMAP models
- two modes for dealing with images with same name:
    - first = keep first image with the same name
    - both = keep both images by renaming them
"""


import os
import shutil
import copy
import argparse
import numpy as np
import pycolmap


parser = argparse.ArgumentParser(description="")
parser.add_argument("--input_colmap_dirs", type=str, help="Input directory with COLMAP model subdirectories")
parser.add_argument("--output_colmap_dir", type=str, help="Output COLMAP model directory")
parser.add_argument("--input_img_dirs", type=str, help="Input directory with image subdirectories")
parser.add_argument("--output_img_dir", type=str, help="Output image directory")
parser.add_argument("--mode", type=str, help="Mode for dealing with images with same name", default="both", choices=["first", "both"])


def main(args):
    if args.input_colmap_dirs is not None:
        assert os.path.isdir(args.input_colmap_dirs)
        assert args.output_colmap_dir is not None
        assert os.path.isdir(args.output_colmap_dir)
    if args.input_img_dirs is not None:
        assert os.path.isdir(args.input_img_dirs)
        assert args.output_img_dir is not None
        assert os.path.isdir(args.output_img_dir)

    img_list = []

    if args.input_colmap_dirs is not None:
        print("- reading COLMAP models")
        models = []

        in_colmap_list = os.listdir(args.input_colmap_dirs)
        in_colmap_list.sort()

        for subdir in in_colmap_list:
            submodel_path = os.path.join(args.input_colmap_dirs, subdir)
            if os.path.isdir(submodel_path):
                submodel = pycolmap.Reconstruction(submodel_path)
                img_list.append([img.name for img in submodel.images.values()])
                models.append(submodel)
    
    if args.input_img_dirs is not None:
        print("- checking images")
        img_subdirs = []

        in_img_list = os.listdir(args.input_img_dirs)
        in_img_list.sort()
        for subdir_idx, subdir in enumerate(in_img_list):
            subimg_path = os.path.join(args.input_img_dirs, subdir)
            img_subdirs.append(subimg_path)
            subimg_list = os.listdir(subimg_path)

            subimg_list = [img_name for img_name in subimg_list if img_name not in img_list[subdir_idx]]
            img_list[subdir_idx] += subimg_list
    
    print("- deal with images with the same name")
    img_names_all = flatten_list(img_list)
    img_names_unique = list(set(img_names_all))
    img_names_counts = [img_names_all.count(name) for name in img_names_unique]
    img_names_mult = [img_names_unique[j] for j in [i for i, x in enumerate(img_names_counts) if x > 1]]

    img_names_map = []
    img_names_single = []
    for sub_idx, submodel in enumerate(img_list):
        sub_map = {}
        for img_name in submodel:
            if img_name not in img_names_mult:
                sub_map[img_name] = img_name
            else:
                if args.mode == "first":
                    if img_name not in img_names_single:
                        sub_map[img_name] = img_name
                        img_names_single.append(img_name)
                    else:
                        sub_map[img_name] = None
                elif args.mode == "both":
                    sub_map[img_name] = img_name + "_{:0>3d}".format(sub_idx)
        img_names_map.append(sub_map)
    
    if args.output_colmap_dir is not None:
        print("- merging COLMAP models")
        merged_model = merge_models(models, img_names_map)

        print("- writing COLMAP model")
        merged_model.write(args.output_colmap_dir)

    if args.output_img_dir is not None:
        print("- copying images")
        for sub_idx, submodel in enumerate(img_list):
            subimg_path = img_subdirs[sub_idx]

            for img_name in submodel.keys():
                new_img_name = img_names_map[sub_idx][img_name]

                old_img_path = os.path.join(subimg_path, img_name)
                new_img_path = os.path.join(args.output_img_dir, new_img_name)
            
                shutil.copyfile(old_img_path, new_img_path)
        

def merge_models(models, img_names_map):
    merged_model = pycolmap.Reconstruction()
    max_cam_id = 1
    max_img_id = 1
    for model_idx, model in enumerate(models):
        cam_id_map = {}
        img_id_map = {}    
        for cam in model.cameras.values():
            merged_cam = cmp_to_cams_list(cam, merged_model.cameras.values())
            if merged_cam is not None:
                cam_id_map[cam.camera_id] = merged_cam.camera_id
            else:
                cam_id_map[cam.camera_id] = max_cam_id
                cam.camera_id = max_cam_id
                max_cam_id += 1
                merged_model.add_camera(cam)

        for img in model.images.values():
            if img_names_map[model_idx][img.name] is not None:
                img_new = pycolmap.Image()
                img_new.name = img_names_map[model_idx][img.name]
                print("C")
                print("{} --> {}".format(img.camera_id, cam_id_map[img.camera_id]))
                # Does not work due to bug in pycolmap (https://github.com/colmap/pycolmap/issues/154)
                img_new.camera_id = cam_id_map[img.camera_id]
                img_cam = merged_model.cameras[cam_id_map[img.camera_id]]
                img_new.set_up(img_cam)
                print("D")
                img_new.image_id = max_img_id
                img_id_map[img.image_id] = max_img_id
                max_img_id += 1
                img_new.points2D = img.points2D # TODO: fix 3D point indices
                img_new.qvec = img.qvec
                img_new.tvec = img.tvec
                
                
                print("---")
                print(img.camera_id)
                
                print("A")
                merged_model.add_image(img_new)
                print("B")
        for pnt3_id, pnt3 in model.points3D.items():
            track_new = pycolmap.Track()
            for te in pnt3.track:
                if te.image_id not in img_id_map:
                    continue

                new_te = copy.deepcopy(te)
                new_te.image_id = img_id_map[te.image_id]
                track_new.add_element(new_te)

            pnt3.track = track_new
            merged_model.add_point3D(pnt3)

    for img_id in merged_model.images.keys():
        merged_model.register_image(img_id)
    
    return merged_model


def cmp_to_cams_list(cam, cams_list):
    for cam2 in cams_list:
        if cmp_cams(cam, cam2):
            return cam2
    return None


def cmp_cams(cam1, cam2):
    if cam1.model_name != cam2.model_name:
        return False
    if cam1.width != cam2.width:
        return False
    if cam1.height != cam2.height:
        return False
    if np.all(cam1.params != cam2.params):
        return False
    return True


def flatten_list(l):
    return [item for sublist in l for item in sublist]


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)