#!/usr/bin/env python
#
# Print commands to run simulations of CFR data.

import os
import argparse


def main(fit_name, model_names, n_rep=1):
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    data_file = os.path.join(study_dir, 'cfr', 'cfr_eeg_mixed.csv')
    patterns_file = os.path.join(study_dir, 'cfr', 'cfr_patterns.hdf5')

    for name in model_names:
        fit_dir = os.path.join(study_dir, 'cfr', 'fits', fit_name, 'cmr_' + name)
        if not os.path.exists(fit_dir):
            raise IOError(f'Fit directory does not exist: {fit_dir}')
        print(f'cfr_sim_cmr.py {data_file} {patterns_file} {fit_dir} -r {n_rep}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fit_name')
    parser.add_argument('model_names')
    parser.add_argument('--n-rep', '-r', type=int, default=1)
    args = parser.parse_args()

    main(args.fit_name, args.model_names.split(','), args.n_rep)
