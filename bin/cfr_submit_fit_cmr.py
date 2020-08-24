#!/usr/bin/env python
#
# Print commands to run fitting of CFR data.

import os
import argparse
import numpy as np


def main(fcf_features, ff_features, sublayers, res_dir, n_rep=10, n_job=48,
         tol=0.00001, n_sim_rep=50):
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    data_file = os.path.join(study_dir, 'cfr', 'cfr_eeg_mixed.csv')
    patterns_file = os.path.join(study_dir, 'cfr', 'cfr_patterns.hdf5')
    inputs = f'{data_file} {patterns_file}'
    opts = f'-t {tol:.6f} -n {n_rep} -j {n_job} -r {n_sim_rep}'

    if sublayers:
        opts = f'-s {opts}'
        res_name = 'cmr-sl'
    else:
        res_name = 'cmr'

    if fcf_features and fcf_features != 'none':
        res_name += f'_fcf-{fcf_features}'
    if ff_features and ff_features != 'none':
        res_name += f'_ff-{ff_features}'
    full_dir = os.path.join(study_dir, 'cfr', res_dir, res_name)

    print(f'cfr_fit_cmr.py {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fcf_features')
    parser.add_argument('ff_features')
    parser.add_argument('res_dir')
    parser.add_argument('--sublayers', '-s', action='store_true')
    parser.add_argument('--n-rep', '-n', default=10, type=int)
    parser.add_argument('--n_job', '-j', default=48, type=int)
    parser.add_argument('--tol', '-t', type=float, default=0.00001)
    parser.add_argument('--n-sim-rep', '-r', type=int, default=1)
    args = parser.parse_args()

    fcf_list = args.fcf_features.split(',')
    ff_list = args.ff_features.split(',')

    max_n = np.max([len(arg) for arg in [fcf_list, ff_list]])
    if len(fcf_list) == 1:
        fcf_list *= max_n
    if len(ff_list) == 1:
        ff_list *= max_n

    for fcf, ff in zip(fcf_list, ff_list):
        main(fcf, ff, args.sublayers, args.res_dir, args.n_rep, args.n_job,
             args.tol, args.n_sim_rep)
