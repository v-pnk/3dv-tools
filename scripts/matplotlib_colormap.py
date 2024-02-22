#!/usr/bin/env python3

"""
Print out (or visualize) samples from a selcted matplotlib colormap.
"""


import argparse
import numpy as np
import matplotlib.cm


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "colormap",
    type=str,
    help="Colormap name - see: https://matplotlib.org/stable/tutorials/colors/colormaps.html",
)
parser.add_argument(
    "--uniform_num", type=int, default=10, help="Number of uniformly sampled values"
)
parser.add_argument(
    "--values", type=float, nargs="+", help="List of values in <0,1> interval"
)
parser.add_argument(
    "--join_format",
    type=str,
    choices=["space", "comma", "np"],
    default="comma",
    help="Number of uniformly sampled values",
)
parser.add_argument("--show", action="store_true", help="Show the colors in CLI")


def main(args):
    if args.uniform_num is not None and args.values is None:
        args.values = np.linspace(0, 1, args.uniform_num)
    color_list = matplotlib.cm.get_cmap(args.colormap)(args.values)[np.newaxis, :, :3]
    for row in range(color_list.shape[1]):
        color = color_list[0, row, :].flatten()

        if args.join_format == "space":
            print("{:.3f} {:.3f} {:.3f}".format(color[0], color[1], color[2]))
        elif args.join_format == "comma":
            print("{:.3f}, {:.3f}, {:.3f}".format(color[0], color[1], color[2]))
        elif args.join_format == "np":
            if row == 0:
                print("[[{:.3f}, {:.3f}, {:.3f}],".format(color[0], color[1], color[2]))
            elif row == color_list.shape[1] - 1:
                print(" [{:.3f}, {:.3f}, {:.3f}]]".format(color[0], color[1], color[2]))
            else:
                print(" [{:.3f}, {:.3f}, {:.3f}],".format(color[0], color[1], color[2]))

    if args.show:
        visualize_colormap(color_list)


def visualize_colormap(color_list):
    colormap_text = ""
    color_strings = []
    colormap_vals = np.linspace(0, 1, 32)

    for row in range(color_list.shape[1]):
        color = color_list[0, row, :].flatten()
        col_i = (255.0 * color).astype(np.uint8)
        color_strings.append(
            "\x1b[38;2;{};{};{}m".format(col_i[0], col_i[1], col_i[2])
            + "\u2588"
            + "\033[0m"
        )

    colormap_text = "min " + "".join(color_strings) + " max"
    print(colormap_text)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
