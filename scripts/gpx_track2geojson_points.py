#!/usr/bin/env python3


"""
Load GPX track file and export GeoJSON with individual track points, named by
the timestamps.

"""


import argparse
import datetime
import xml.etree.ElementTree as ET
import json
import numpy as np


parser = argparse.ArgumentParser(
    description="Load GPX track file and export GeoJSON with individual track points, named by the timestamps."
)
parser.add_argument("gpx_file", type=str, help="The input GPX track file.")
parser.add_argument("geojson_output", type=str, help="The output GeoJSON file.")


def main(args):
    timestamps, coords_wgs84 = load_gpx_file(args.gpx_file)

    geojson = {
        "type": "FeatureCollection",
        "features": [],
    }

    for time, coord in zip(timestamps, coords_wgs84.T):
        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [coord[1], coord[0], coord[2]],
            },
            "properties": {
                "name": time.strftime("%H:%M:%S.%f"),
            },
        }
        geojson["features"].append(feature)

    with open(args.geojson_output, "wt") as f:
        json.dump(geojson, f)


def load_gpx_file(gpx_file: str):
    """Parse the GPX file and get the time series of GNSS coordinates.

    Parameters:
    gpx_file (str): The path to the GPX file.

    Returns:
    timestamps (np.ndarray): The time series of the GNSS coordinates in UTC.
    coords_wgs84 (np.ndarray): The GNSS coordinates.

    """

    timestamps = np.empty((0,), dtype=datetime.datetime)
    coords_wgs84 = np.empty((3, 0))

    tree = ET.parse(gpx_file)
    root = tree.getroot()
    schema = root.attrib[
        "{http://www.w3.org/2001/XMLSchema-instance}schemaLocation"
    ].split()[0]

    for trkpt in root.findall(".//{" + schema + "}trkpt"):
        lat = float(trkpt.attrib["lat"])
        lon = float(trkpt.attrib["lon"])

        alt = float(trkpt.find("{" + schema + "}ele").text)

        # The time in GPX should be in UTC / GMT time zone
        time_str = trkpt.find("{" + schema + "}time").text
        time_str_crop = time_str[:19]  # crop the subsecond part and Z
        time = datetime.datetime.strptime(time_str_crop, "%Y-%m-%dT%H:%M:%S")
        if len(time_str) > 20:
            # Add the microseconds
            time = time.replace(microsecond=1000 * int(time_str[20:23]))

        timestamps = np.append(timestamps, time)
        coords_wgs84 = np.append(coords_wgs84, np.array([[lat], [lon], [alt]]), axis=1)

    return timestamps, coords_wgs84


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
