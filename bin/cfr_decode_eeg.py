#!/usr/bin/env python
#
# Decode CFR EEG patterns.

import os
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from cfr import task
from cfr import decode


def decode_subject(patterns_dir, subject, clf='svm'):
    pattern_file = os.path.join(patterns_dir, f'sub-{subject}_pattern.txt')
    events_file = os.path.join(patterns_dir, f'sub-{subject}_events.csv')
    pattern = np.loadtxt(pattern_file)
    events = pd.read_csv(events_file)

    pattern = decode.impute_samples(pattern)
    evidence = decode.classify_patterns(events, pattern, clf=clf)

    out_file = os.path.join(patterns_dir, f'sub-{subject}_decode.csv')
    df = pd.concat([events, evidence], axis=1)
    df.to_csv(out_file)


def main(patterns_dir, n_jobs=1, subjects=None, clf=None):
    if subjects is None:
        subjects, _ = task.get_subjects()

    Parallel(n_jobs=n_jobs)(
        delayed(decode_subject)(patterns_dir, subject, clf) for subject in subjects
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'patterns_dir', help="Path to directory with patterns to decode."
    )
    parser.add_argument(
        '--n-jobs', '-n', type=int, default=1,
        help="Number of processes to run in parallel."
    )
    parser.add_argument(
        '--subjects', '-s', default=None, help="Comma-separated list of subjects."
    )
    parser.add_argument(
        '--classifier', '-c', default='svm', help="classifier type."
    )
    args = parser.parse_args()

    if args.subjects is not None:
        inc_subjects = [f'LTP{subject:0>3}' for subject in args.subjects.split(',')]
    else:
        inc_subjects = None
    main(args.patterns_dir, args.n_jobs, inc_subjects, args.classifier)
