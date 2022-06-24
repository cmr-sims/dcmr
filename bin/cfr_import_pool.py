#!/usr/bin/env python
#
# Import the pool from CFR to standard format.


import argparse
import numpy as np
import pandas as pd
from cfr import task


def main(image_dir, map_file, output_dir, pool_file):
    pool = task.read_pool_cfr(image_dir)
    task.save_pool_images(pool, output_dir)
    pool = pool.drop(columns=['filepath'])

    # sort to match map order
    item_map = pd.read_csv(map_file)
    pool_sorted = pool.copy()
    for i, item in item_map['item'].iteritems():
        pool_ind = pool['item'].str.upper().to_numpy() == item.upper()
        pool_sorted.iloc[i, :] = pool.iloc[np.where(pool_ind)[0][0], :]
    pool_sorted.to_csv(pool_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Write a pool information spreadsheet and a standardized image directory."
    )
    parser.add_argument('image_dir', help="Path to directory with pool images")
    parser.add_argument('map_file', help="Path to pool map file")
    parser.add_argument('output_dir', help="Path to output directory")
    parser.add_argument('pool_file', help="Path to output pool CSV file")
    args = parser.parse_args()
    main(args.image_dir, args.map_file, args.output_dir, args.pool_file)
