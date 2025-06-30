#!/usr/bin/env python3


"""
Generate a set of distinct colors.

This script generates a set of distinct colors in the RGB color space using
distinctipy library (https://github.com/alan-turing-institute/distinctipy).
"""


import argparse
import distinctipy


parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "n",
    type=int,
    help="Number of distinct colors to generate",
)
parser.add_argument(
    "--init_colors",
    type=str,
    help="Initial colors which will be taken into account during the generation"
         "(RGB values (8-bit int or float) divided by commas, or 6-digit hex codes, "
         "individual colors separated by semicolons, e.g.: '255,0,0; 0.0,0.12,0.0; 0000FF')",
)
parser.add_argument(
    "--output_format",
    type=str,
    choices=["hex", "rgb_float", "rgb_int"],
    default="rgb_float",
    help="Output format of the generated colors",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed for random number generator",
)


def main(args):
    
    init_colors = parse_input_colors(args.init_colors)
    colors = distinctipy.get_colors(
        n_colors=args.n,
        exclude_colors=init_colors,
        return_excluded=False,
        rng=args.seed,
    )

    for color in colors:
        red_int = int(color[0] * 255)
        green_int = int(color[1] * 255)
        blue_int = int(color[2] * 255)
        print(
            "\x1b[38;2;{};{};{}m".format(red_int, green_int, blue_int) + 4 * "\u2588" + "\033[0m ", end="",
            )
        if args.output_format == "hex":
            hex_color = "#{:02x}{:02x}{:02x}".format(
                red_int, green_int, blue_int
            )
            print(hex_color)
        elif args.output_format == "rgb_float":
            rgb_float = "{:.2f}, {:.2f}, {:.2f}".format(
                color[0], color[1], color[2]
            )
            print(rgb_float)
        elif args.output_format == "rgb_int":
            rgb_int = "{}, {}, {}".format(
                red_int, green_int, blue_int
            )
            print(rgb_int)


def parse_input_colors(init_colors):
    """
    Parse the input colors from a string, returing a list of RGB float tuples.
    """
    colors = []
    if init_colors:
        for color in init_colors.split(";"):
            color = color.replace(" ", "")
            if "," in color:
                # RGB values
                rgb_values = [float(x.strip()) for x in color.split(",")]
                if len(rgb_values) == 3:
                    r, g, b = rgb_values
                    # try to recognize the format RGB format
                    if ((r > 1 or g > 1 or b > 1) and 
                        ((r % 1) < 1e-6 and (g % 1) < 1e-6 and (b % 1) < 1e-6)):
                        # the color is represented as an 8-bit int
                        r, g, b = r / 255, g / 255, b / 255
                else:
                    raise ValueError(f'Invalid RGB format: "{color}"')
            else:
                # Hex code
                if color.startswith("#"):
                    hex_str = color[1:]
                else:
                    hex_str = color
                
                if len(hex_str) != 6:
                    raise ValueError(f'Invalid hex code: "{color}"')
                    
                r = int(hex_str[:2], 16) / 255
                g = int(hex_str[2:4], 16) / 255
                b = int(hex_str[4:], 16) / 255
            
            colors.append((r, g, b))
                
    return colors



if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
