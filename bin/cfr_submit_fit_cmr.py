#!/usr/bin/env python
#
# Print commands to run fitting of CFR data.

import os
import argparse
import numpy as np


def main(
    fcf_features,
    ff_features,
    sublayers,
    res_dir,
    subpar,
    fixed,
    n_rep=10,
    n_job=48,
    tol=0.00001,
    n_sim_rep=50,
):
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    data_file = os.path.join(study_dir, 'cfr', 'cfr_eeg_mixed.csv')
    patterns_file = os.path.join(study_dir, 'cfr', 'cfr_patterns.hdf5')
    inputs = f'{data_file} {patterns_file}'
    opts = f'-t {tol:.6f} -n {n_rep} -j {n_job} -r {n_sim_rep}'

    if sublayers:
        opts = f'-s {opts}'
        res_name = 'cmrs'
    else:
        res_name = 'cmr'

    if fcf_features and fcf_features != 'none':
        res_name += f'_fcf-{fcf_features}'
    if ff_features and ff_features != 'none':
        res_name += f'_ff-{ff_features}'
    if subpar:
        opts += f' -p {subpar}'
        res_name += f'_sl-{subpar}'
    if fixed:
        opts += f' -f {fixed}'
        res_name += f'_fix-{fixed.replace("=", "")}'
    full_dir = os.path.join(study_dir, 'cfr', res_dir, res_name)

    print(f'cfr_fit_cmr.py {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fit multiple models to a dataset. See cfr_fit_cmr.py for details.'
    )
    parser.add_argument('fcf_features')
    parser.add_argument('ff_features')
    parser.add_argument('res_dir')
    parser.add_argument('--sublayers', '-s', action='store_true')
    parser.add_argument('--sublayer-param', '-p', default=None)
    parser.add_argument('--fixed', '-f', default=None)
    parser.add_argument('--n-rep', '-n', default=10, type=int)
    parser.add_argument('--n_job', '-j', default=48, type=int)
    parser.add_argument('--tol', '-t', type=float, default=0.00001)
    parser.add_argument('--n-sim-rep', '-r', type=int, default=1)
    args = parser.parse_args()

    fcf_list = args.fcf_features.split(',')
    ff_list = args.ff_features.split(',')
    if args.sublayer_param is not None:
        sub_list = args.sublayer_param.split(',')
    else:
        sub_list = []
    if args.fixed is not None:
        fix_list = args.fixed.split(',')
    else:
        fix_list = []

    max_n = np.max([len(arg) for arg in [fcf_list, ff_list, sub_list, fix_list]])
    if len(fcf_list) == 1:
        fcf_list *= max_n
    if len(ff_list) == 1:
        ff_list *= max_n
    if args.sublayer_param is not None:
        if len(sub_list) == 1:
            sub_list *= max_n
    else:
        sub_list = [None] * max_n
    if args.fixed is not None:
        if len(fix_list) == 1:
            fix_list *= max_n
    else:
        fix_list = [None] * max_n

    for fcf, ff, sub, fix in zip(fcf_list, ff_list, sub_list, fix_list):
        main(
            fcf,
            ff,
            args.sublayers,
            args.res_dir,
            sub,
            fix,
            args.n_rep,
            args.n_job,
            args.tol,
            args.n_sim_rep,
        )
