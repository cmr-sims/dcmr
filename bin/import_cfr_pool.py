#!/usr/bin/env python
#
# Import the pool from CFR to standard format.


import argparse
from cfr import task


def main(image_dir, output_dir, pool_file):
    pool = task.read_pool_cfr(image_dir)
    task.save_pool_images(pool, output_dir)
    pool = pool.drop(columns=['filepath'])
    pool.to_csv(pool_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image_dir', help="Path to directory with pool images")
    parser.add_argument('output_dir', help="Path to output directory")
    parser.add_argument('pool_file', help="Path to output pool CSV file")
    args = parser.parse_args()
    main(args.image_dir, args.output_dir, args.pool_file)
