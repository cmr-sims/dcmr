#!/usr/bin/env python
#
# Decode CFR simulated context patterns.

from pathlib import Path
import logging
import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from psifr import fr
from cymr import cmr
from cfr import task
from cfr import decode
from cfr import framework


def decode_subject(
    data_file,
    patterns_file,
    fit_dir,
    eeg_class_dir,
    sublayer,
    out_dir,
    subject,
    **kwargs,
):
    """Decode category from simulated context patterns for one suject."""
    log_dir = out_dir / 'logs'
    log_dir.mkdir(exist_ok=True)
    subject_id = f'LTP{subject:03n}'
    log_file = log_dir / f'sub-{subject_id}_log.txt'

    logger = logging.getLogger(subject_id)
    formatter = logging.Formatter('%(asctime)s %(levelname)s:%(name)s:%(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    logger.info(f'Loading data from {data_file}.')
    data = task.read_study_recall(data_file)
    study = fr.filter_data(data, subjects=subject, trial_type='study').reset_index()
    study = decode.mark_included_eeg_events(
        study, eeg_class_dir, subjects=[subject_id]
    )

    param_file = fit_dir / 'fit.csv'
    logger.info(f'Loading parameters from {param_file}.')
    subj_param = framework.read_fit_param(param_file)

    config_file = fit_dir / 'parameters.json'
    logger.info(f'Loading model configuration from {config_file}.')
    param_def = cmr.read_config(config_file)

    logger.info(f'Loading model patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    logger.info('Recording network states.')
    model = cmr.CMR()
    state = model.record(
        study, {}, subj_param, param_def=param_def, patterns=patterns, include=['c']
    )
    net = state[0]
    context = np.vstack([s.c[net.get_slice('c', sublayer, 'item')] for s in state])

    logger.info(f'Running classification.')
    include = study['include'].to_numpy()
    study_include = study.loc[include]
    evidence = decode.classify_patterns(
        study_include, context[include], logger=logger, **kwargs
    )
    evidence.index = study_include.index

    out_file = out_dir / f'sub-{subject_id}_decode.csv'
    logger.info(f'Writing results to {out_file}.')
    df = pd.concat([study, evidence], axis=1)
    df.to_csv(out_file.as_posix())


def main(
    data_file,
    patterns_file,
    fit_dir,
    eeg_class_dir,
    sublayer,
    res_name,
    n_jobs=1,
    subjects=None,
    **kwargs,
):
    if subjects is None:
        _, subjects = task.get_subjects()

    out_dir = fit_dir / f'decode_{sublayer}' / res_name
    out_dir.mkdir(exist_ok=True, parents=True)
    Parallel(n_jobs=n_jobs)(
        delayed(decode_subject)(
            data_file,
            patterns_file,
            fit_dir,
            eeg_class_dir,
            sublayer,
            out_dir,
            subject,
            **kwargs,
        )
        for subject in subjects
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Decode category from EEG patterns measured during the CFR study."
    )
    parser.add_argument('data_file', help='path to Psifr csv data file')
    parser.add_argument('patterns_file', help='path to network patterns')
    parser.add_argument(
        'fit_dir', help="Path to directory with model fit results."
    )
    parser.add_argument(
        'eeg_class_dir', help="Path to directory with EEG classification results."
    )
    parser.add_argument('sublayer', help='sublayer of context to decode.')
    parser.add_argument('res_name', help='name of results subdirectory.')
    parser.add_argument(
        '--n-jobs',
        '-n',
        type=int,
        default=1,
        help="Number of processes to run in parallel.",
    )
    parser.add_argument(
        '--subjects', '-s', default=None, help="Comma-separated list of subjects."
    )
    parser.add_argument(
        '--normalization',
        '-l',
        default='range',
        help='Normalization to apply before classification {"z", "range"}',
    )
    parser.add_argument(
        '--classifier',
        '-c',
        default='svm',
        help='classifier type {"svm", "logreg", "plogreg"}',
    )
    parser.add_argument(
        '--multi-class',
        '-m',
        default='auto',
        help='multi-class method {"auto", "ovr", "multinomial"}',
    )
    parser.add_argument(
        '-C', type=float, default=1, help='Regularization parameter (1.0)'
    )
    args = parser.parse_args()

    if args.subjects is not None:
        inc_subjects = [int(subject) for subject in args.subjects.split(',')]
    else:
        inc_subjects = None
    main(
        Path(args.data_file),
        Path(args.patterns_file),
        Path(args.fit_dir),
        Path(args.eeg_class_dir),
        args.sublayer,
        args.res_name,
        args.n_jobs,
        inc_subjects,
        normalization=args.normalization,
        clf=args.classifier,
        multi_class=args.multi_class,
        C=args.C,
    )
