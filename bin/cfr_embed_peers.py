#!/usr/bin/env python
#
# Embed PEERS words as vectors.

import argparse


def main(data_file, out_file, sem_file):
    import sys
    import numpy as np
    from scipy import stats
    import pandas as pd
    from psifr import fr
    from cymr import network
    try:
        import tensorflow_hub as hub
    except ModuleNotFoundError:
        print("Error: TensorflowHub must be installed to run embedding.")
        sys.exit(1)

    # get item pool
    data = pd.read_csv(data_file)
    study = fr.filter_data(data, trial_type="study")
    items = np.sort(study["item"].unique()).tolist()

    # run embedding
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    patterns = embed(items).numpy()

    # save semantic representation file
    np.savez(sem_file, items=items, vectors=patterns)

    # localist, category, and distributional patterns
    loc_patterns = np.eye(len(items))
    cat_patterns = np.ones((len(items), 1))
    use_z = stats.zscore(patterns, axis=1) / np.sqrt(patterns.shape[1])

    # write patterns to standard format hdf5 file
    network.save_patterns(out_file, items, loc=loc_patterns, cat=cat_patterns, use=use_z)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed PEERS words as vectors")
    parser.add_argument("data_file", help="Path to PEERS data file in CSV format.")
    parser.add_argument("patterns_file", help="Path to HDF5 file to save patterns.")
    parser.add_argument("sem_file", help="Path to NPZ semantic similarity file.")
    args = parser.parse_args()
    main(args.data_file, args.patterns_file, args.sem_file)
