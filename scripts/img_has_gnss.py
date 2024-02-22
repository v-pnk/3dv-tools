#!/usr/bin/env python3

"""
Check if the given images have GNSS information in their metadata.
"""


import os
import argparse
import exiftool


parser = argparse.ArgumentParser(description="")
parser.add_argument("image_dir", type=str, help="The directory with images")
parser.add_argument(
    "--write_info",
    type=str,
    choices=["all", "positive", "negative"],
    default="all",
    help="Which cases to print out",
)

GREEN = "\033[92m"
RED = "\033[91m"
END = "\033[0m"


def main(args):
    file_list = [
        os.path.join(args.image_dir, file_i) for file_i in os.listdir(args.image_dir)
    ]
    file_list.sort()

    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(file_list)

    for meta_i, file_i in zip(metadata, file_list):
        name_i = os.path.basename(file_i)
        if ("EXIF:GPSLatitude" in meta_i.keys()) and (
            "EXIF:GPSLongitude" in meta_i.keys()
        ):
            latitude = float(meta_i["EXIF:GPSLatitude"])
            longitude = float(meta_i["EXIF:GPSLongitude"])

            if args.write_info in ["all", "positive"]:
                print(name_i)
                print(
                    GREEN
                    + "{:.6f}{} {:.6f}{}".format(
                        latitude,
                        meta_i["EXIF:GPSLatitudeRef"],
                        longitude,
                        meta_i["EXIF:GPSLongitudeRef"],
                    )
                    + END
                )
        else:
            if args.write_info in ["all", "negative"]:
                print(name_i)
                print(RED + "NO GNSS INFO" + END)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
