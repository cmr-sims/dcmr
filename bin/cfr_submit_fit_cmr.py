#!/usr/bin/env python
#
# Print commands to run fitting of CFR data.

import os
import argparse
import numpy as np


def main(fcf_features, ff_features, n_rep=10, n_job=29, tol=0.00001):
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    data_file = os.path.join(study_dir, 'cfr', 'cfr_eeg_mixed.csv')
    patterns_file = os.path.join(study_dir, 'cfr', 'cfr_patterns.hdf5')
    inputs = f'{data_file} {patterns_file}'
    opts = f'-t {tol:.6f} -n {n_rep} -j {n_job}'

    res_name = 'cmr'
    if fcf_features and fcf_features != 'none':
        res_name += f'_fcf-{fcf_features}'
    if ff_features and ff_features != 'none':
        res_name += f'_ff-{ff_features}'
    res_dir = os.path.join(study_dir, 'cfr', 'fits', res_name)

    print(f'cfr_fit_cmr.py {inputs} {fcf_features} {ff_features} {res_dir} {opts}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fcf_features')
    parser.add_argument('ff_features')
    parser.add_argument('--n-rep', '-n', default=10, type=int)
    parser.add_argument('--n_job', '-j', default=29, type=int)
    parser.add_argument('--tol', '-t', type=float, default=0.00001)
    args = parser.parse_args()

    fcf_list = args.fcf_features.split(',')
    ff_list = args.ff_features.split(',')

    max_n = np.max([len(arg) for arg in [fcf_list, ff_list]])
    if len(fcf_list) == 1:
        fcf_list *= max_n
    if len(ff_list) == 1:
        ff_list *= max_n

    for fcf, ff in zip(fcf_list, ff_list):
        main(fcf, ff, args.n_rep, args.n_job, args.tol)
