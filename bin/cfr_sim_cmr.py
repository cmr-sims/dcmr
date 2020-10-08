#!/usr/bin/env python
#
# Simulate data based on subject parameters.

import os
import argparse
import pandas as pd
from cymr import cmr
from cymr import network
from cymr import parameters
from cfr import framework


def main(data_file, patterns_file, fit_dir, n_rep=1):

    if not os.path.exists(fit_dir):
        raise IOError(f"Fit directory does not exist: {fit_dir}")

    # load trials to simulate
    data = pd.read_csv(data_file)
    study_data = data.loc[(data['trial_type'] == 'study')]

    # get model, patterns, and weights
    model = cmr.CMRDistributed()
    patterns = network.load_patterns(patterns_file)
    param_file = os.path.join(fit_dir, 'parameters.json')
    param_def = parameters.read_json(param_file)

    # load parameters
    fit_file = os.path.join(fit_dir, 'fit.csv')
    subj_param = framework.read_fit_param(fit_file)

    # run simulation
    sim = model.generate(
        study_data, {}, subj_param, param_def, patterns, n_rep=n_rep
    )

    # save
    sim_file = os.path.join(fit_dir, 'sim.csv')
    sim.to_csv(sim_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('patterns_file', help="Path to HDF5 file with patterns")
    parser.add_argument('fit_dir')
    parser.add_argument('--n-rep', '-r', type=int, default=1)
    args = parser.parse_args()

    main(args.data_file, args.patterns_file, args.fit_dir, args.n_rep)
