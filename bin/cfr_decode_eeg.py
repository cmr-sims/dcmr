#!/usr/bin/env python
#
# Decode CFR EEG patterns.

from pathlib import Path
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from cfr import task
from cfr import decode


def decode_subject(patterns_dir, out_dir, subject, **kwargs):
    pattern_file = patterns_dir / f'sub-{subject}_pattern.txt'
    events_file = patterns_dir / f'sub-{subject}_events.csv'
    pattern = np.loadtxt(pattern_file.as_posix())
    events = pd.read_csv(events_file)

    pattern = decode.impute_samples(pattern)
    evidence = decode.classify_patterns(events, pattern, **kwargs)

    out_file = out_dir / f'sub-{subject}_decode.csv'
    df = pd.concat([events, evidence], axis=1)
    df.to_csv(out_file.as_posix())


def main(patterns_dir, out_dir, n_jobs=1, subjects=None, **kwargs):
    if subjects is None:
        subjects, _ = task.get_subjects()

    out_dir.mkdir(exist_ok=True)
    Parallel(n_jobs=n_jobs)(
        delayed(decode_subject)(
            patterns_dir, out_dir, subject, **kwargs
        ) for subject in subjects
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'patterns_dir', help="Path to directory with patterns to decode."
    )
    parser.add_argument(
        'out_dir', help="Path to output directory."
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
    parser.add_argument(
        '--multi-class', '-m', default='auto', help='multi-class method {"auto", "ovr", "multinomial"}'
    )
    parser.add_argument(
        '-C', type=float, default=1, help='Regularization parameter (1.0)'
    )
    args = parser.parse_args()

    if args.subjects is not None:
        inc_subjects = [f'LTP{subject:0>3}' for subject in args.subjects.split(',')]
    else:
        inc_subjects = None
    main(
        Path(args.patterns_dir),
        Path(args.out_dir),
        args.n_jobs,
        inc_subjects,
        clf=args.classifier,
        multi_class=args.multi_class,
        C=args.C
    )
