#!/usr/bin/env python3


"""
Tool for parsing SRT files generated by DJI drones. The SRT file (subtitle
file) contains metadata for each frame of the video, including GNSS coordinates
and camera settings.

The tool was tested with SRT files generated by Mini 4 Pro. Other drones and 
firmware versions may produce SRT files in different formats.

The tool is reusing and adjusting the code first published in:
https://github.com/v-pnk/long-img-org

"""


import os
from datetime import datetime, timedelta
import re
import math
import json
import argparse
import xml.etree.ElementTree as ET
from tqdm import tqdm

import numpy as np
import cv2
import exiftool


parser = argparse.ArgumentParser()
parser.add_argument(
    "dji_srt_path", 
    type=str, 
    help="Path to the input SRT file"
)
parser.add_argument(
    "--video_path", 
    type=str, 
    help="Path to the input video file"
)
parser.add_argument(
    "--geojson_path",
    type=str,
    help="Path to the output GeoJSON file with the flight path / frame locations",
)
parser.add_argument(
    "--gpx_path",
    type=str,
    help="Path to the output GPX file with the flight path / frame locations",
)
parser.add_argument(
    "--frame_dir", type=str, help="Path to the output directory with video frames"
)

parser.add_argument(
    "--frame_rate",
    type=float,
    default=1.0,
    help="Frame rate for the extraction (frames per second)",
)
parser.add_argument(
    "--res_ratio",
    type=float,
    default=1.0,
    help="Resolution ratio used for downsampling the frames - can be in (0.0 - 1.0) range",
)
parser.add_argument(
    "--image_eq",
    type=str,
    choices=["off", "global", "clahe"],
    default="off",
    help="Mode of histogram equalization of the extracted frames - off / global / clahe",
)

parser.add_argument(
    "--geo_mode",
    type=str,
    default="only_frames",
    choices=["full", "only_frames"],
    help="Which data to include in the GeoJSON / GPX file - full log from the SRT file or only the positions of the extracted video frames",
)
parser.add_argument(
    "--geo_write_mode",
    type=str,
    default="a",
    choices=["a", "w"],
    help="If the output GeoJSON / GPX file exists, append (a) or overwrite it (w)",
)


def main(args):
    print("Parsing the SRT file...")
    timestamps, coords_wgs84, frame_metadata = load_dji_srt_file(args.dji_srt_path)

    if args.video_path is not None:
        assert os.path.exists(args.video_path), "The given video file does not exist: {}".format(args.video_path)

        print("Parsing video metadata...")
        video_metadata = get_metadata(args.video_path)
        frame_idx_list = []

        with exiftool.ExifToolHelper() as et:
            print("Extracting the frames and writing to the output directory...")
            for frame, idx in tqdm(extract_frames(args.video_path, 
                                                  frame_rate=args.frame_rate, 
                                                  res_ratio=args.res_ratio)):

                frame_idx_list.append(idx)
                image_name = time_to_name(timestamps[idx])
                image_path = os.path.join(args.frame_dir, image_name + ".jpg")
                
                if args.image_eq != "off":
                    frame["image"] = equalize_img(frame["image"], args.image_eq)

                cv2.imwrite(image_path, frame["image"])
                del frame["image"]

                exif_tags = {
                    "XResolution": video_metadata["x_resolution"],
                    "YResolution": video_metadata["y_resolution"],
                    "ImageWidth": round(video_metadata["width"] * args.res_ratio),
                    "ImageHeight": round(video_metadata["height"] * args.res_ratio),
                }

                exif_tags.update(time_to_exif(timestamps[idx]))
                exif_tags.update(WGS84_to_exif(coords_wgs84[:, idx]))

                if "iso" in frame_metadata[idx]["metadata"]:
                    exif_tags["EXIF:ISO"] = frame_metadata[idx]["metadata"]["iso"]
                if "shutter" in frame_metadata[idx]["metadata"]:
                    exif_tags["EXIF:ShutterSpeedValue"] = sec_frac_to_apex(
                        frame_metadata[idx]["metadata"]["shutter"]
                    )
                if "fnum" in frame_metadata[idx]["metadata"]:
                    exif_tags["EXIF:FNumber"] = frame_metadata[idx]["metadata"]["fnum"]
                # TODO: Is there an EXIF tag corresponding to EV (Exposure Value)?
                # TODO: What is the focal length value in the SRT file?

                et.set_tags(image_path, exif_tags, params=["-overwrite_original"])
    else:
        frame_idx_list = []
        ideal_capture_time = timestamps[0]
        frame_period = timedelta(seconds=1.0 / args.frame_rate)

        for frame_idx, this_time in enumerate(timestamps):
            next_time = this_time + frame_period

            if abs((this_time - ideal_capture_time).total_seconds()) < abs((next_time - ideal_capture_time).total_seconds()):
                frame_idx_list.append(frame_idx)
                ideal_capture_time += frame_period


    if args.geojson_path is not None:
        if args.geo_mode == "full":
            print("Writing the full location log to the GeoJSON file...")
            write_geojson(args.geojson_path, coords_wgs84, timestamps)
        else:
            print("Writing the frame locations to the GeoJSON file...")
            write_geojson(
                args.geojson_path,
                coords_wgs84[:, frame_idx_list],
                timestamps[frame_idx_list],
            )

    if args.gpx_path is not None:
        if args.geo_mode == "full":
            print("Writing the full location log to the GPX file...")
            write_gpx(args.gpx_path, coords_wgs84, timestamps)
        else:
            print("Writing the frame locations to the GPX file...")
            write_gpx(
                args.gpx_path,
                coords_wgs84[:, frame_idx_list],
                timestamps[frame_idx_list],
            )


def load_dji_srt_file(dji_srt_path):
    """
    Load metadata from a DJI SRT file.

    Parameters:
    dji_srt_path (str): Path to the DJI SRT file.

    Returns:
    timestamps (list): List of timestamps.
    coords_wgs84 (list): List of WGS84 coordinates.
    srt_data (list): List of dictionaries with the SRT data.
    """

    assert os.path.exists(dji_srt_path), "The given file does not exist"
    assert dji_srt_path.endswith(
        (".srt", ".SRT")
    ), "The file must have the .srt extension"

    timestamps = np.empty((0,), dtype=datetime)
    coords_wgs84 = np.empty((3, 0))
    frame_metadata = []

    with open(dji_srt_path, "rt") as f:
        lines = f.readlines()

    for i in range(0, len(lines), 6):
        frame_counter_str = lines[i + 2].strip()
        frame_datetime_str = lines[i + 3].strip()
        frame_metadata_str = lines[i + 4].strip()

        metadata = {}

        # Line with frame counter
        # Example: "<font size="28">FrameCnt: 8, DiffTime: 33ms"
        frame_counter = int(frame_counter_str.split(",")[0].split(":")[1])

        # Line with date and time
        # Example: "2024-01-20 16:43:19.563"
        frame_datetime = datetime.strptime(frame_datetime_str, "%Y-%m-%d %H:%M:%S.%f")

        # Line with frame metadata
        # Example: "[iso: 100] [shutter: 1/100.0] [fnum: 1.7] [ev: 0] [color_md: dlog_m] [focal_len: 20.30] [latitude: 50.073279] [longitude: 14.336132] [rel_alt: 6.600 abs_alt: 207.861] [ct: 4500] </font>"
        # - color_md = color mode
        # - rel_alt = relative altitude to the home point
        # - abs_alt = absolute altitude above sea level
        # - ct = color temperature
        metadata_substrs = re.findall(r"\[.*?\]", frame_metadata_str)

        for substr in metadata_substrs:
            words = re.split(": | ", substr)
            metadata.update(
                {
                    words[i].replace("[", ""): words[i + 1].replace("]", "")
                    for i in range(0, len(words), 2)
                }
            )

        timestamps = np.append(timestamps, frame_datetime)
        coords_wgs84 = np.c_[
            coords_wgs84,
            [
                float(metadata["latitude"]),
                float(metadata["longitude"]),
                float(metadata["rel_alt"]),
            ],
        ]
        frame_metadata.append(
            {
                "frame_counter": frame_counter,
                "frame_datetime": frame_datetime,
                "metadata": metadata,
            }
        )

    return timestamps, coords_wgs84, frame_metadata


def extract_frames(video_path, frame_rate=1, res_ratio=1.0):
    """
    Extract frames with valid capture times from a video file.

    Parameters:
    video (str): Video file.
    frame_rate (float): Frame rate for the extraction (frames per second).


    Yields:
    data (dict): Data dictionary containing the frame and original size.
    frame_idx (int): Frame index.
    """

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    frame_rate = min(cap.get(cv2.CAP_PROP_FPS), frame_rate)

    ideal_capture_time = 0.0
    success, frame = cap.read()

    frame_idx = 0
    while success:
        this_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        next_time = this_time + 1.0 / cap.get(cv2.CAP_PROP_FPS)

        if abs(this_time - ideal_capture_time) < abs(next_time - ideal_capture_time):
            data = {}

            # Downsample the frame
            if res_ratio < 1.0:
                frame_resized = cv2.resize(
                    frame,
                    (0, 0),
                    fx=res_ratio,
                    fy=res_ratio,
                    interpolation=cv2.INTER_AREA,
                )
            else:
                frame_resized = frame

            data["image"] = frame_resized
            data["orig_width"] = frame.shape[1]
            data["orig_height"] = frame.shape[0]

            # frames.append(data)
            ideal_capture_time += 1.0 / frame_rate

            yield data, frame_idx

        success, frame = cap.read()
        frame_idx += 1

    cap.release()


def get_metadata(video_path):
    """
    Get selected metadata of a video file.

    Parameters:
    video_path (str): Video file path.

    Returns:
    metadata (dict): Selected metadata of the video file.
    """

    metadata = {}

    with exiftool.ExifToolHelper() as et:
        exif_metadata = et.get_metadata(video_path)[0]

    metadata["width"] = exif_metadata["QuickTime:ImageWidth"]
    metadata["height"] = exif_metadata["QuickTime:ImageHeight"]
    metadata["x_resolution"] = exif_metadata["QuickTime:XResolution"]
    metadata["y_resolution"] = exif_metadata["QuickTime:YResolution"]
    metadata["bit_depth"] = exif_metadata["QuickTime:BitDepth"]

    return metadata


def time_to_name(full_datetime: datetime):
    """Get the image name from the creation date and time. The image name is in
    the following format:
    <year>-<month>-<day>_<hour>-<minute>-<second>-<millisecond>

    Parameters:
    full_datetime (datetime): The full date and time of the image capture.

    Returns:
    image_name (str): The image name.

    """

    datetime_str = full_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    millis_str = "{:03d}".format(int(0.001 * full_datetime.microsecond))
    image_name = "{}-{}".format(datetime_str, millis_str)

    return image_name


def time_to_exif(full_datetime: datetime):
    """Convert the full date and time to EXIF tags.

    Parameters:
    full_datetime (datetime): The full date and time of the image capture.

    Returns:
    exif_tags (dict): The EXIF date and time tags.

    """

    capture_datetime = full_datetime.strftime("%Y:%m:%d %H:%M:%S")
    capture_millisecond = full_datetime.strftime("%f")[:3]
    capture_tz_offset = full_datetime.strftime("%z")
    capture_tz_offset = capture_tz_offset[:3] + ":" + capture_tz_offset[3:]

    exif_tags = {
        "EXIF:DateTimeOriginal": capture_datetime,
        "EXIF:SubSecTimeOriginal": capture_millisecond,
        "EXIF:OffsetTimeOriginal": capture_tz_offset,
    }

    return exif_tags


def WGS84_to_exif(coords: np.ndarray):
    """Convert GNSS coordinates to EXIF tags.

    Parameters:
    coords (np.ndarray): The GNSS coordinates with size (2,) or (3,).

    Returns:
    exif_tags (dict): The EXIF GPS-related tags.

    """

    coords = coords.flatten()

    lat = coords[0]
    lon = coords[1]

    lat_ref = "N"
    lon_ref = "E"

    if lat < 0:
        lat_ref = "S"
        lat = -lat

    if lon < 0:
        lon_ref = "W"
        lon = -lon

    exif_tags = {
        "EXIF:GPSLatitude": lat,
        "EXIF:GPSLatitudeRef": lat_ref,
        "EXIF:GPSLongitude": lon,
        "EXIF:GPSLongitudeRef": lon_ref,
    }

    if coords.shape[0] == 2:
        alt = coords[2]
        alt_ref = 0

        if alt < 0:
            alt_ref = 1
            alt = -alt

        exif_tags["EXIF:GPSAltitude"] = alt
        exif_tags["EXIF:GPSAltitudeRef"] = alt_ref

    return exif_tags


def sec_frac_to_apex(sec_frac: str):
    """Convert seconds to the APEX time format.

    Parameters:
    sec_frac (str): The time in fractions of seconds.

    Returns:
    apex_time (float): The time in the APEX format.

    """

    if "/" in sec_frac:
        frac = sec_frac.split("/")
        seconds = float(frac[0]) / float(frac[1])
    else:
        seconds = float(sec_frac)

    return -math.log2(seconds)


def equalize_img(img, mode):
    """Equalize the histogram of the image.

    Parameters:
    img (np.ndarray): The input image.
    mode (str): Mode of histogram equalization - simple global histogram 
        equalization or CLAHE

    Returns:
    img_eq (np.ndarray): The image with equalized histogram.

    """

    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # equalize the histogram of the luma component

    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    elif mode == "global":
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    else:
        raise ValueError("Invalid mode: {}".format(mode))

    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)

    return img_eq


def write_geojson(file_path, coords_wgs84, timestamps, write_mode="a"):
    """Write the location points to a GeoJSON file.

    Parameters:
    file_path (str): The path to the output GeoJSON file.
    coords_wgs84 (np.ndarray): The WGS84 coordinates.
    timestamps (list): The timestamps.

    """

    if write_mode == "a" and os.path.exists(file_path):
        with open(file_path, "rt") as f:
            geojson_data = json.load(f)
    else:
        geojson_data = {
            "type": "FeatureCollection",
            "features": [],
        }

    for i in range(len(timestamps)):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coords_wgs84[1, i], coords_wgs84[0, i]],
            },
            "properties": {
                "datetime": timestamps[i].strftime("%Y-%m-%d %H:%M:%S"),
                "marker-size": "small",
            },
        }

        geojson_data["features"].append(feature)

    # Create line between the points
    linestring = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [coords_wgs84[1, i], coords_wgs84[0, i]] for i in range(len(timestamps))
            ],
        },
        "properties": {
            "stroke": "#000000",
            "stroke-width": 1,
        },
    }
    geojson_data["features"].append(linestring)

    # Write the GeoJSON file
    with open(file_path, "wt") as f:
        json.dump(geojson_data, f, indent=4)


def write_gpx(file_path, coords_wgs84, timestamps, write_mode="a"):
    """Write the location points to a GPX file.

    Parameters:
    file_path (str): The path to the output GPX file.
    coords_wgs84 (np.ndarray): The WGS84 coordinates.
    timestamps (list): The timestamps.

    """

    if write_mode == "a" and os.path.exists(file_path):
        tree = ET.parse(file_path)
        trk = tree.find("trk")
    else:
        gpx = ET.Element(
            "gpx",
            attrib={
                "version": "1.1",
                "creator": "3DV-Tools: DJI SRT Tools - https://github.com/v-pnk/3dv-tools",
            },
        )
        tree = ET.ElementTree(gpx)
        trk = ET.SubElement(gpx, "trk")
    
    trkseg = ET.SubElement(trk, "trkseg")

    for i in range(len(timestamps)):
        trkpt = ET.SubElement(
            trkseg,
            "trkpt",
            attrib={"lat": str(coords_wgs84[0, i]), "lon": str(coords_wgs84[1, i])},
        )
        ET.SubElement(trkpt, "time").text = (
            timestamps[i].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        )
        ET.SubElement(trkpt, "ele").text = str(coords_wgs84[2, i])

    tree.write(file_path, encoding="UTF-8", xml_declaration=True)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
