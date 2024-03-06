#!/usr/bin/env python3


"""
Export the EXIF GNSS locations from a given image set to a GeoJSON file
- the generated file can be viewed, e.g., in MapBox (geojson.io web app)
  - supports custom marker color and size using Mapbox Simple Style

- color modes
  - none: all points are black
  - timestamp: color assigned linearly between oldest to the newest timestamp
  - month: color assigned based on the month of the year
"""


import os
import argparse
import numpy as np
import exiftool
import json
from datetime import datetime
from tqdm import tqdm


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "image_dir", 
    type=str, 
    help="The directory with images (works recursively)"
)
parser.add_argument(
    "geojson_file", 
    type=str, 
    help="Path to the GeoJSON file"    
)
parser.add_argument(
    "--color_mode",
    type=str,
    choices=["timestamp", "month"],
    default="timestamp",
    help="Mode of coloring of the points",
)
parser.add_argument(
    "--filter_month",
    type=str,
    help="Filter the images by month (1-12) and only include those in the GeoJSON file. Range can also be specified (e.g. 12-2 for December to February)",
)


def main(args):
    print("EXIF GNSS to GeoJSON")
    img_metadata_all = {}

    valid_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    file_list = [
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(args.image_dir)
        for f in filenames
        if os.path.splitext(f)[1].lower() in valid_exts
    ]

    print("- reading EXIF data from images")
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(file_list)

    print("- collecting EXIF data")
    for meta_i, file_i in zip(metadata, file_list):
        name_i = os.path.relpath(file_i, args.image_dir)

        if ("EXIF:GPSLatitude" in meta_i.keys()) and (
            "EXIF:GPSLongitude" in meta_i.keys()
        ):
            latitude = float(meta_i["EXIF:GPSLatitude"])
            longitude = float(meta_i["EXIF:GPSLongitude"])

            img_metadata_all[name_i] = {
                "latitude": latitude,
                "longitude": longitude,
                "color": (0, 0, 0),
            }
        else:
            print("  - no GNSS data in: {}".format(name_i))
            continue

        if "EXIF:DateTimeOriginal" in meta_i.keys():
            dt = datetime.strptime(meta_i["EXIF:DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            img_metadata_all[name_i]["datetime"] = dt

    # Filter the images by month
    if args.filter_month is not None:
        if "-" in args.filter_month:
            month_start, month_end = [int(m) for m in args.filter_month.split("-")]
        else:
            month_start = month_end = int(args.filter_month)

        if month_start > month_end:
            months = list(range(month_start, 13)) + list(range(1, month_end + 1))
        else:
            months = list(range(month_start, month_end + 1))

        img_metadata_all = {
            key: img_metadata_all[key]
            for key in img_metadata_all.keys()
            if img_metadata_all[key]["datetime"].month in months
        }

        if len(img_metadata_all) == 0:
            print("No images found for the specified month range")
            return
        else:
            print("Found {} images for the specified month range".format(len(img_metadata_all)))

    print("- coloring points")
    if args.color_mode == "timestamp":
        # - normalize the timestamps to [0,1] range
        all_timestamps = [
            img_metadata_all[key]["datetime"].timestamp()
            for key in img_metadata_all.keys()
            if "datetime" in img_metadata_all[key].keys()
        ]
        total_timespan = max(all_timestamps) - min(all_timestamps)

        # - create a colormap
        import matplotlib.cm

        for img_name in img_metadata_all.keys():
            if "datetime" in img_metadata_all[img_name].keys():
                rel_timestamp = (
                    img_metadata_all[img_name]["datetime"].timestamp()
                    - min(all_timestamps)
                ) / total_timespan
                color = list(matplotlib.cm.get_cmap("rainbow")(rel_timestamp)[:3])
                img_metadata_all[img_name]["color"] = tuple(
                    [int(255 * c) for c in color]
                )

    elif args.color_mode == "month":
        cmap = month_colormap()
        for img_name in img_metadata_all.keys():
            if "datetime" in img_metadata_all[img_name].keys():
                month = img_metadata_all[img_name]["datetime"].month
                color = list(cmap[:, month - 1])
                img_metadata_all[img_name]["color"] = tuple(
                    [int(255 * c) for c in color]
                )

        visualize_colormap(cmap, 3, 1)
        print("  JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC")

    # Create the GeoJSON data structure
    print("- writing GeoJSON file")
    geojson_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": name_i,
                    "datetime": img_metadata_all[name_i]["datetime"].strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if "datetime" in img_metadata_all[name_i].keys()
                    else None,
                    "marker-color": "#%02x%02x%02x" % img_metadata_all[name_i]["color"],
                    "marker-size": "small",
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        img_metadata_all[name_i]["longitude"],
                        img_metadata_all[name_i]["latitude"],
                    ],
                },
            }
            for name_i in img_metadata_all.keys()
        ],
    }

    # Write the GeoJSON file
    with open(args.geojson_file, "wt") as f:
        json.dump(geojson_data, f, indent=4)


def month_colormap():
    return (
        np.array(
            [
                [88, 199, 234],
                [20, 120, 199],
                [154, 251, 144],
                [22, 118, 28],
                [183, 82, 217],
                [242, 60, 60],
                [242, 130, 60],
                [255, 198, 66],
                [199, 159, 130],
                [145, 95, 55],
                [100, 98, 96],
                [220, 240, 240],
            ]
        ).T / 255.0
    )


def visualize_colormap(color_list, rep_num=1, space_num=0):
    np.repeat(color_list, rep_num, axis=1)
    color_strings = ["  "]

    for row in range(color_list.shape[1]):
        color = color_list[:, row].flatten()
        col_i = (255.0 * color).astype(np.uint8)
        col_str = (
            "\x1b[38;2;{};{};{}m".format(col_i[0], col_i[1], col_i[2])
            + "\u2588"
            + "\033[0m"
        ) * rep_num
        color_strings.append(col_str)
        color_strings.append(" " * space_num)

    print("".join(color_strings))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
