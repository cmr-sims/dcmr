#!/usr/bin/env python
#
# Fit CMR to CFR data.

import argparse
import pandas as pd
from cymr import models
from cymr import network
from cfr import framework


def main(data_file, patterns_file, model_type, sem_model, res_file, n_jobs=1):

    data = pd.read_csv(data_file)
    model = models.CMRDistributed()
    wp = framework.model_variant(model_type, sem_model)
    patterns = network.load_patterns(patterns_file)
    results = model.fit_indiv(data, wp.fixed, wp.free, wp.dependent,
                              patterns=patterns, weights=wp.weights,
                              n_jobs=n_jobs, method='de')
    results.to_csv(res_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('patterns_file')
    parser.add_argument('model_type')
    parser.add_argument('sem_model')
    parser.add_argument('res_file')
    parser.add_argument('--n-jobs', '-j', type=int, default=1)
    args = parser.parse_args()
    main(args.data_file, args.patterns_file, args.model_type, args.sem_model,
         args.res_file, args.n_jobs)
