#!/usr/bin/env python


"""
Visualize COLMAP model with rerun library
"""


import os
from datetime import datetime, timedelta
import atexit
import numpy as np
from PIL import Image
import argparse
import rerun as rr
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument('colmap_dir', type=str, help='Path to the COLMAP model directory')
parser.add_argument('--img_dir', type=str, help='Path to the images')

parser.add_argument('--vis_cams', action='store_true', help='Visualize cameras')
parser.add_argument('--vis_imgs', action='store_true', help='Visualize images - needs --vis_cams')
parser.add_argument('--vis_2d_pnts', action='store_true', help='Visualize 2D points')
parser.add_argument('--vis_3d_pnts', action='store_true', help='Visualize 3D points')

parser.add_argument('--img_exif_time', action='store_true', help='Log EXIF time for images')

parser.add_argument('--mesh', type=str, help='Path to a mesh file - not implemented yet in rerun')


def main(args):
    if args.vis_imgs:
        assert args.vis_cams, "WARN: --vis_imgs requires --vis_cams"

    colmap_model = pycolmap.Reconstruction(args.colmap_dir)

    rr.init("rerun_colmap_visualizer", spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN)

    # log mesh
    if args.mesh is not None:
        assert os.path.exists(args.mesh), "WARN: mesh file does not exist"
        assert os.path.splitext(args.mesh)[1] in [".obj", ".glb"], "WARN: mesh file must be .obj or .glb"
        rr.log("world/mesh", rr.Asset3D(path=args.mesh))


    for img in colmap_model.images.values():
        cam = colmap_model.cameras[img.camera_id]
        img_basename = os.path.splitext(img.name)[0]

        if args.img_dir is not None:
            img_path = os.path.join(args.img_dir, img.name)

        # log images
        img_pil = None
        if args.img_dir is not None and os.path.exists(img_path):
            img_pil = Image.open(img_path)
            
            # log EXIF time for images
            if args.img_exif_time:    
                exif_data = img_pil._getexif()
                if exif_data and 36867 in exif_data:
                    # get Exif.Image.DateTimeOriginal tag (36867) from image file
                    exif_time = datetime.strptime(exif_data[36867], "%Y:%m:%d %H:%M:%S")
                    if 37521 in exif_data:
                        # get Exif.Photo.SubSecTimeOriginal tag (37521) for sub-second precision
                        # the units of Exif.Photo.SubSecTimeOriginal depend on the number of digits
                        exif_us = exif_data[37521].ljust(6, '0')
                        exif_time = exif_time.replace(microsecond=int(exif_us))

                    epoch = datetime.utcfromtimestamp(0)
                    nanos_since_epoch = int((exif_time - epoch) / timedelta(microseconds=1) * 1e3)
                    rr.set_time_nanos("timeline", nanos_since_epoch, recording=None) 
                else:
                    print("WARN: Image EXIF does not contain \"DateTimeOriginal\" tag - {}".format(img_path))

        # log 2D points
        if args.vis_2d_pnts and img.points2D is not None and img.points2D:
            uv = []
            for pnt2D in img.points2D:
                uv.append(pnt2D.xy)
            rr.log(
                "world/cameras/{}/image/keypoints".format(img_basename),
                rr.Points2D(uv, colors=[252, 54, 54]),
            )

        # log cameras
        if args.vis_cams:
            # log camera poses
            q = img.qvec
            q = np.array([q[1], q[2], q[3], q[0]])

            rr.log(
                "world/cameras/{}".format(img_basename), 
                rr.Transform3D(translation=img.tvec, rotation=rr.Quaternion(xyzw=q), from_parent=True)
                )

            if len(cam.focal_length_idxs()) == 1:
                fx = cam.focal_length
                fy = cam.focal_length
            else:
                fx = cam.focal_length_x
                fy = cam.focal_length_y
            
            # log camera intrinsics
            rr.log(
                "world/cameras/{}/image".format(img_basename),
                rr.Pinhole(
                    width=cam.width,
                    height=cam.height,
                    focal_length=(fx, fy),
                    principal_point=(cam.principal_point_x, cam.principal_point_y)
                )
            )

        # log the image here (after logging the camera) to prevent GUI scaling issues
        if args.vis_imgs:
            assert args.img_dir is not None, "WARN: --vis_imgs requires --img_dir"
            rr.log(
                "world/cameras/{}/image".format(img_basename),
                rr.Image(np.array(img_pil))
            )

        # log 3D points
        if args.vis_3d_pnts:
            pnt3D_xyz = []
            pnt3D_rgb = []
            pnt3D_error = []
            pnt3D_track = []
            for pnt3D in colmap_model.points3D.values():
                pnt3D_xyz.append(pnt3D.xyz)
                pnt3D_rgb.append(pnt3D.color)
                pnt3D_error.append(pnt3D.error)
                pnt3D_track.append(pnt3D.track.length())
        
            assert pnt3D_xyz, "WARN: No 3D points to visualize"
            rr.log(
                "world/points3D",
                rr.Points3D(pnt3D_xyz, colors=pnt3D_rgb), 
                rr.AnyValues(error=pnt3D_error, track=pnt3D_track) # add error and track length as attributes for each point
            )
            

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)