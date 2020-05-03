#!/usr/bin/env python
#
# Fit CMR to CFR data.

import os
import argparse
import pandas as pd
from cymr import models


def main(data_file, res_dir, n_jobs=1):

    data = pd.read_csv(data_file)
    model = models.CMR()
    fixed = {'B_rec': .8, 'L': 1, 'T': 10, 'X1': .05, 'X2': 1}
    var_names = ['B_enc', 'B_rec']
    var_bounds = {'B_enc': (0, 1), 'B_rec': (0, 1)}
    results = model.fit_indiv(data, fixed, var_names, var_bounds, n_jobs=n_jobs)

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    res_file = os.path.join(res_dir, 'cfr_cmr_fit.csv')
    results.to_csv(res_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('res_dir')
    parser.add_argument('--n-jobs', '-j', type=int, default=1)
    args = parser.parse_args()
    main(args.data_file, args.res_dir, args.n_jobs)
