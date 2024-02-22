#!/usr/bin/env python3


"""
Add Gaussian noise to depth maps
"""


import argparse
import os
import numpy as np
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('input_dir', type=str, help='path to the input directory with depth maps')
parser.add_argument('output_dir', type=str, help='path to the input directory with depth maps')
parser.add_argument('--noise_type', type=str, default='additive', choices=['additive', 'multiplicative'], help='type of the noise (additive or multiplicative)')
parser.add_argument('--noise_stddev', type=float, default=0.1, help='Standa of the Gaussian noise (in depth units)')


def main(args):
    in_files = os.listdir(args.input_dir)
    in_files = [f for f in in_files if f.endswith('.npy') or f.endswith('.npz')]
    in_files.sort()

    for in_file in tqdm(in_files):
        in_path = os.path.join(args.input_dir, in_file)
        out_path = os.path.join(args.output_dir, in_file)

        if in_file.endswith('.npy'):
            depth_map = np.load(in_path).astype(np.float32)
        elif in_file.endswith('.npz'):
            depth_map = np.load(in_path)['depth'].astype(np.float32)
        
        noise = np.random.normal(0, args.noise_sigma, depth_map.shape)

        if args.noise_type == 'additive':
            depth_map += noise
        elif args.noise_type == 'multiplicative':
            depth_map *= (1 + noise)

        depth_map[depth_map < 0] = 0

        if in_file.endswith('.npy'):
            np.save(out_path, depth_map.astype(np.float16))
        elif in_file.endswith('.npz'):
            np.savez_compressed(out_path, depth=depth_map.astype(np.float16))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
