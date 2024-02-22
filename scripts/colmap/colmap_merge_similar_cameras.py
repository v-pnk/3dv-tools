#!/usr/bin/env python3


"""
Reduce number of cameras in COLMAP model by merging similar ones.
"""


import os
import math
import argparse
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True,
                    help="Path to the input COLMAP model")
parser.add_argument("--output_path", type=str, required=True,
                    help="Path to the output COLMAP model")
parser.add_argument("--output_type", type=str, required=False,
                    choices=["BIN", "TXT"], default="BIN",
                    help="Type of the output model")
parser.add_argument("--large_diff_thr", type=float, default=100, 
                    help="Difference threshold for large values > 1 (fx, fy, cx, cy)")
parser.add_argument("--small_diff_thr", type=float, default=0.1, 
                    help="Difference threshold for small values < 1 (k, p)")


def main(args):
    print("COLMAP model camera merger")

    print("- Loading input COLMAP model")
    input_colmap_model = pycolmap.Reconstruction(args.input_path)

    all_cam_list = []
    sim_list = []
    unq_cam_list = []

    print("- Searching for similar cameras")
    for _, cam in input_colmap_model.cameras.items():
        sim_idx = check_similar(cam, unq_cam_list, args.large_diff_thr, args.small_diff_thr)

        if sim_idx is None:
            sim_list.append(len(unq_cam_list))
            unq_cam_list.append(cam)
        else:
            sim_list.append(sim_idx)

        all_cam_list.append(cam)
    
    print("- Merging similar cameras")
    id_map, new_cam_list = mean_cam(sim_list, all_cam_list)

    print("- Writing output COLMAP model in {} format".format(args.output_type))
    output_colmap_model = pycolmap.Reconstruction()
    for new_cam in new_cam_list:
        output_colmap_model.add_camera(new_cam)

    for img_id, img in input_colmap_model.images.items():
        new_image = pycolmap.Image(
            name=img.name, tvec=img.tvec, qvec=img.qvec, camera_id=id_map[img.camera_id], id=img.image_id)
        output_colmap_model.add_image(new_image)

    for img_id, img in input_colmap_model.images.items():
        output_colmap_model.register_image(img_id)

    if args.output_type == "BIN":
        output_colmap_model.write_binary(args.output_path)
    elif args.output_type == "TXT":
        output_colmap_model.write_text(args.output_path)


def check_similar(cam_c, unq_cam_list, large_diff_thr, small_diff_thr):
    for idx, cam_l in enumerate(unq_cam_list):
        if (cam_l.height != cam_c.height) or (cam_l.width != cam_c.width) or (cam_l.model_name != cam_c.model_name):
            continue
        


        for param_l, param_c in zip(cam_l.params, cam_c.params):
            if param_c <= 1.0:
                abs_diff_thr = small_diff_thr
            else:
                abs_diff_thr = large_diff_thr

            if abs(param_l - param_c) > abs(abs_diff_thr):
                break
        else:
            # - the program counter gets here only if the above loop did not 
            #   break (finished successfully)
            return idx
    
    return None


def mean_cam(sim_list, all_cam_list):
    unq_cam_idx = list(set(sim_list))
    id_map = {}
    new_cam_list = []
    for new_idx, unq_idx in enumerate(unq_cam_idx):
        new_id = new_idx + 1
        sim_cam_list = [b for a,b, in zip(sim_list, all_cam_list) if a == unq_idx]
        params_sum = [0] * len(sim_cam_list[0].params)

        w = sim_cam_list[0].width
        h = sim_cam_list[0].height
        model_name = sim_cam_list[0].model_name

        for sim_cam in sim_cam_list:
            assert w == sim_cam.width
            assert h == sim_cam.height
            assert model_name == sim_cam.model_name

            id_map[sim_cam.camera_id] = new_id
            params_sum = [a+b for a, b in zip(params_sum, sim_cam.params)]
        
        new_params = [a / len(sim_cam_list) for a in params_sum]

        params_std = [0] * len(sim_cam_list[0].params)
        for sim_cam in sim_cam_list:
            id_map[sim_cam.camera_id] = new_id
            params_std = [a + ((b-c)*(b-c)) for a, b, c in zip(params_std, new_params, sim_cam.params)]
        
        params_std = [math.sqrt(a / len(sim_cam_list)) for a in params_std]

        print("  - New camera {}:".format(new_id))
        print("    - merging {} cameras".format(len(sim_cam_list)))
        print("      {}x{} {}".format(w, h, model_name))
        print("    - params avg:")
        print("      {}".format(new_params))
        print("    - params standard deviation:")
        print("      {}".format(params_std))

        new_cam_list.append(pycolmap.Camera(model=sim_cam.model_name, width=sim_cam.width, height=sim_cam.height, params=new_params, id=new_id))

    return (id_map, new_cam_list)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
