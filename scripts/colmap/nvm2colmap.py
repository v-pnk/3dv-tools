#!/usr/bin/env python


"""
Convert NVM (VisualSFM SfM reconstruction) to COLMAP model.
- NVM format: http://ccwu.me/vsfm/doc.html#nvm
"""


import os
import numpy as np
import argparse
import imagesize # pip install imagesize
import glob


parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument('input_nvm', type=str,
                    help='Path to the NVM file')
parser.add_argument('image_dir', type=str,
                    help='Path to the directory with images (or image directories) corresponding to the NVM file (required to get image size)')
parser.add_argument('output_colmap', type=str,
                    help='Path to the output text COLMAP model')
parser.add_argument('--underscores', action="store_true",
                    help='Replace spaces and slashes in image names by underscores')
parser.add_argument('--omit_points', action="store_true",
                    help='Do not extract 3D points, keypoints and matches')


def main(args):
    assert os.path.isfile(args.input_nvm)
    assert os.path.isdir(args.image_dir)
    assert os.path.isdir(args.output_colmap)

    cam_data, pnt3_data = parse_nvm(args.input_nvm)
    write_colmap(args.output_colmap, cam_data, pnt3_data)


def parse_nvm(path):
    print("- parsing NVM file")
    f = open(path, 'r')

    state = "header"

    cam_data = []
    pnt3_data = []

    for line in f:
        line = line.strip()

        if (state == "header") and line:
            continue
        elif (state == "header") and not line:
            state = "cameras"
            print("  - parsing camera data")
            next(f)
            continue
        elif (state == "cameras") and line:
            # <File name> <focal length> <quaternion WXYZ> <camera center> <radial distortion> 0
            words = line.split()
            name = words[0]
            fl = float(words[1])
            qvec = np.array(list(map(float, words[2:6])))
            cvec = np.array(list(map(float, words[6:9])))
            k = float(words[9])

            R = qvec2rotmat(qvec)
            tvec = -R @ cvec

            img_path = os.path.splitext(os.path.join(args.image_dir, name))[0]

            # - find the corresponding file to the file name from the NVM model
            found_path = min(glob.glob(img_path + '*', recursive=True), key=len)

            assert os.path.isfile(found_path), "ERROR: " + found_path + \
                " image path is invalid - check the given path to the image directory"
            w, h = imagesize.get(found_path)
            
            found_name = os.path.join(os.path.dirname(name), os.path.basename(found_path))

            if args.underscores:
                found_name = found_name.replace(" ", "_")
                found_name = found_name.replace("/", "_")

            cam_data.append({"qvec": qvec, "tvec": tvec, "f": fl, "k": k,
                "w" : w, "h" : h, "name" : found_name, "pnt2_data":[]})
            
        elif (state == "cameras") and not line:
            state = "points"

            if args.omit_points:
                break
            
            print("  - parsing 3D point data")

            next(f)
            continue
        elif (state == "points") and line:
            # <XYZ> <RGB> <number of measurements> <List of Measurements>
            # <Measurement> = <Image index> <Feature Index> <xy>
            words = line.split()
            xyz = np.array(list(map(float, words[0:3])))
            rgb = np.array(list(map(int, words[3:6])))
            meas_num = int(words[6])
            measurements = words[7 : (7 + 4*meas_num)]
            

            track_data = []
            for i in range(meas_num):
                meas = measurements[4*i : 4*(i+1)]
                img_idx = int(meas[0])

                # - disregard the feature indices
                #   - many features are unused and do not transfer to 
                #     the COLMAP model (we don't have any info about unused 
                #     features in NVM file)
                # feat_idx = int(meas[1])

                # - the keypoints ("measurements") are defined relative to the 
                #   principle point
                kpt_rel_c = np.array(list(map(float, meas[2:4])))

                # - save 2D points corresponding to the 3D point
                track_data.append({"img_id": img_idx, "pnt2_id": len(cam_data[img_idx]["pnt2_data"])})
                # - add 2D point data and 3D point ID to corresponding images
                cam_data[img_idx]["pnt2_data"].append({"pnt3_id": len(pnt3_data), "kpt_rel_c": kpt_rel_c})
                
            pnt3_data.append({"xyz": xyz, "rgb": rgb, "track": track_data})
                
        elif (state == "points") and not line:
            break
        else:
            continue

    f.close()

    return cam_data, pnt3_data


def write_colmap(path, cam_data, pnt3_data):
    print("- generating COLMAP TXT model")

    images_file_out=os.path.join(args.output_colmap, "images.txt")
    cameras_file_out=os.path.join(path, "cameras.txt")
    points3D_file_out=os.path.join(path, "points3D.txt")

    cam_num = len(cam_data)
    obs_num = 0

    f_cam = open(cameras_file_out, "w")
    print("  - writing cameras.txt")
    f_cam.write("# Camera list with one line of data per camera:\n")
    f_cam.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    f_cam.write("# Number of cameras: {}\n".format(cam_num))

    for img_id, data in enumerate(cam_data):
        obs_num += len(cam_data[img_id]["pnt2_data"])

        name = data["name"]
        tvec = data["tvec"]
        qvec = data["qvec"]

        # - NVM uses 0 based indexing, COLMAP 1 based indexing
        # - the distortion in COLMAP is applied to the projections and not to 
        #   the measurements --> (-1 * k)

        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # SIMPLE_RADIAL model: f, cx, cy, k
        cam_line = "{} {} {} {} {} {} {} {}\n".format(img_id+1, "SIMPLE_RADIAL", data["w"], data["h"], data["f"], data["w"]/2, data["h"]/2, -1 * data["k"])
        
        f_cam.write(cam_line)

    f_cam.close()

    # - compute the mean number of observations per image
    mean_obs_per_img = obs_num / len(cam_data)

    f_img=open(images_file_out, "w")
    print("  - writing images.txt")
    f_img.write("# Image list with two lines of data per image:\n")
    f_img.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f_img.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    f_img.write("# Number of images: {}, mean observations per image: {}\n".format(
        cam_num, mean_obs_per_img))

    for img_id, data in enumerate(cam_data):
        name = data["name"]
        tvec = data["tvec"]
        qvec = data["qvec"]

        img_line="{} {} {} {} {} {} {} {} {} {}\n".format(img_id+1,
            qvec[0], qvec[1], qvec[2], qvec[3], tvec[0], tvec[1], tvec[2],
            img_id+1, name)
        
        pnt2_parts=[]
        for pnt2 in data["pnt2_data"]:
            # - the keypoints in NVM are defined relative to the principle point
            #   - NVM does not store the principle point --> half of image size
            kpt = pnt2["kpt_rel_c"] + 0.5 * np.array([data["w"], data["h"]])

            pnt2_parts.append("{} {} {}".format(
                kpt[0], kpt[1], pnt2["pnt3_id"]))
        pnt2_line = ' '.join(pnt2_parts) + '\n'

        f_img.write(img_line)
        f_img.write(pnt2_line)

    f_img.close()

    if pnt3_data:
        track_len_sum = 0
        for pnt3 in pnt3_data:
            track_len_sum += len(pnt3["track"])
        mean_track_len = track_len_sum / len(pnt3)
        
        f_pnt = open(points3D_file_out, "w")
        print("  - writing points3D.txt")
        f_pnt.write("# 3D point list with one line of data per point:\n")
        f_pnt.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f_pnt.write("# Number of points: {}, mean track length: {}\n".format(len(pnt3), mean_track_len))

        for pnt3_id, data in enumerate(pnt3_data):
            xyz = data["xyz"]
            rgb = data["rgb"]
            error = 0 # TODO
            pnt3_line = "{} {} {} {} {} {} {} {}".format(
                pnt3_id, xyz[0], xyz[1], xyz[2], rgb[0], rgb[1], rgb[2], error)
            pnt2_parts = []
            for pnt2 in data["track"]:
                pnt2_parts.append("{} {}".format(pnt2["img_id"]+1, pnt2["pnt2_id"]))
            pnt3_line += ' ' + ' '.join(pnt2_parts) + '\n'

            f_pnt.write(pnt3_line)

        f_pnt.close()
    else:
        f_pnt = open(points3D_file_out, "w")
        f_pnt.close()


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])
    

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
