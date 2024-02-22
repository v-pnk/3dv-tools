#!/usr/bin/env python3


"""
Show a color in the terminal.
"""


import os
import argparse


parser = argparse.ArgumentParser(description="")
parser.add_argument('rgb', type=str, nargs='+', help='RGB values - divided by spaces or commas, or 6-digit hex code')
parser.add_argument('--type', type=str, choices=["float", "int"], default="float",
                    help="input data type - float [0,1], or int [0,255]")


def main(args):
    if len(args.rgb) == 1:
        # Hex code
        if args.rgb[0][0] == '#':
            hex_str = args.rgb[0][1:]
        
        assert len(rgb) == 6, "Invalid hex code: \"{}\"".format(hex_str)
        red = int(hex_str[:2], 16)
        green = int(hex_str[2:4], 16)
        blue = int(hex_str[4:], 16)
    else:
        # RGB values
        rgb = ' '.join(args.rgb)

        if ',' in rgb:
            red, green, blue = rgb.split(',')
        else:
            red, green, blue = rgb.split(' ')
        
        red = float(red.strip())
        green = float(green.strip())
        blue = float(blue.strip())

        if red > 1 or green > 1 or blue > 1:
            args.type = "int"

        if args.type == "float":
            red = int(red*255)
            green = int(green*255)
            blue = int(blue*255)
        elif args.type == "int":
            red = int(red)
            green = int(green)
            blue = int(blue)
    cli_len = os.get_terminal_size()[0]
    print("\x1b[38;2;{};{};{}m".format(red, green, blue) + cli_len*"\u2588" + "\033[0m")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)