#!/usr/bin/env python
#
# Record CFR CMR patterns.

import os
import argparse
from joblib import Parallel, delayed
import json
import numpy as np
import pandas as pd

from cymr import cmr
from cfr import task


def record_subject(data, subj_param, subject, param_def, patterns):
    model = cmr.CMR()
    subj_data = data.query(f'subject == {subject}').copy()
    state = model.record(
        subj_data,
        subj_param[subject],
        param_def=param_def,
        patterns=patterns,
        include=['c', 'c_in'],
    )
    c = np.array([s.c for s in state])
    c_in = np.array([s.c_in for s in state])

    net = state[0]
    sublayers = list(net.c_segment.keys())
    ind = {
        sublayer: [int(i) for i in net.get_segment('c', sublayer, 'item')]
        for sublayer in sublayers
    }
    net_record = {'c': c, 'c_in': c_in}
    record = {'record': net_record, 'ind': ind, 'data': subj_data}
    return record


def main(model_dir, model_version, model_name, n_jobs=1, subjects=None):
    if subjects is None:
        _, subjects = task.get_subjects()

    # paths to input files
    pattern_file = os.path.join(model_dir, 'cfr_patterns.hdf5')
    data_file = os.path.join(model_dir, 'cfr_eeg_mixed.csv')
    fit_dir = os.path.join(model_dir, 'fits', model_version, model_name)
    param_file = os.path.join(fit_dir, 'parameters.json')
    fit_file = os.path.join(fit_dir, 'fit.csv')

    # load best-fitting parameters
    param_def = cmr.read_config(param_file)
    results = pd.read_csv(fit_file, index_col=0)
    subj_param = results.T.to_dict()

    # prepare simulation
    patterns = cmr.load_patterns(pattern_file)
    data = pd.read_csv(data_file)
    labeled = task.label_clean_trials(data)
    clean = labeled.query('clean').reset_index()
    clean = clean.loc[clean['subject'].isin(subjects)].copy()

    # record = record_subject(clean, subj_param, 1, param_def, patterns)
    record = Parallel(n_jobs=n_jobs)(
        delayed(record_subject)(clean, subj_param, subject, param_def, patterns)
        for subject in subjects
    )

    # save recording to fit directory
    record_dir = os.path.join(fit_dir, 'record')
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    # save a text file for each array
    arrays = {}
    for key in record[0]['record'].keys():
        arr = np.vstack([rec['record'][key] for rec in record])
        arrays[key] = arr
    np.savez(os.path.join(record_dir, 'state.npz'), **arrays)

    # save the data subset that only includes clean events
    clean.to_csv(os.path.join(record_dir, 'data.csv'))

    # save network indices for context segments
    with open(os.path.join(record_dir, 'indices.json'), 'w') as f:
        json.dump(record[0]['ind'], f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a simulation and record states of context."
    )
    parser.add_argument('model_dir', help="Path to main model directory.")
    parser.add_argument('model_version', help="Version of the fit.")
    parser.add_argument('model_name', help="Model name.")
    parser.add_argument(
        '--n-jobs',
        '-n',
        type=int,
        default=1,
        help="Number of processes to run in parallel.",
    )
    parser.add_argument(
        '--subjects',
        '-s',
        default=None,
        help="Comma-separated list of subject numbers.",
    )
    args = parser.parse_args()

    if args.subjects is not None:
        inc_subjects = [int(s) for s in args.subjects.split(',')]
    else:
        inc_subjects = None
    main(args.model_dir, args.model_version, args.model_name, args.n_jobs, inc_subjects)
