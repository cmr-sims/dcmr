#!/usr/bin/env python
#
# Print commands to run fitting of CFR data.

import os
import argparse
import numpy as np


def main(model_types, sem_models, res_names, n_rep=10, n_job=29):
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    data_file = os.path.join(study_dir, 'cfr', 'cfr_eeg_mixed.csv')
    patterns_file = os.path.join(study_dir, 'cfr', 'cfr_patterns.hdf5')
    inputs = f'{data_file} {patterns_file}'
    opts = f'-n {n_rep} -j {n_job}'
    for model, sem, res in zip(model_types, sem_models, res_names):
        res_file = os.path.join(study_dir, 'cfr', 'fits', res + '.csv')
        print(f'cfr_fit_cmr.py {inputs} {model} {sem} {res_file} {opts}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_types')
    parser.add_argument('sem_models')
    parser.add_argument('res_names')
    parser.add_argument('--n-rep', '-n', default=10, type=int)
    parser.add_argument('--n_job', '-j', default=29, type=int)
    args = parser.parse_args()

    # unpack specifications
    model_list = args.model_types.split(',')
    sem_list = args.sem_models.split(',')
    res_list = args.res_names.split(',')
    max_n = np.max([len(arg) for arg in [model_list, sem_list, res_list]])
    if len(model_list) == 1:
        model_list *= max_n
    if len(sem_list) == 1:
        sem_list *= max_n
    if len(res_list) == 1:
        res_list *= max_n

    main(model_list, sem_list, res_list, args.n_rep, args.n_job)
