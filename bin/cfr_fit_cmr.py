#!/usr/bin/env python
#
# Fit CMR to CFR data.

import os
import argparse
import pandas as pd
from cymr import cmr
from cymr import network
from cymr import fit
from cfr import framework


def main(data_file, patterns_file, fcf_features, ff_features, sublayers,
         res_dir, sublayer_param=None, fixed_param=None, n_reps=1, n_jobs=1,
         tol=0.00001, n_sim_reps=1, include=None):

    # prepare model for search
    data = pd.read_csv(data_file)
    if include is not None:
        data = data.loc[data['subject'].isin(include)]

    # set parameter definitions based on model framework
    model = cmr.CMRDistributed()
    param_def = framework.model_variant(
        fcf_features, ff_features, sublayers=sublayers,
        sublayer_param=sublayer_param
    )
    patterns = network.load_patterns(patterns_file)

    # fix parameters if specified
    if fixed_param is not None:
        for expr in fixed_param:
            param_name, val = expr.split('=')
            param_def.set_fixed({param_name: float(val)})
            if param_name not in param_def.free:
                raise ValueError(f'Parameter {param_name} is not free.')
            del param_def.free[param_name]

    # save model information
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    json_file = os.path.join(res_dir, 'parameters.json')
    param_def.to_json(json_file)

    # run individual subject fits
    results = model.fit_indiv(
        data, param_def, patterns=patterns, n_jobs=n_jobs, method='de',
        n_rep=n_reps, tol=tol
    )

    # full search information
    res_file = os.path.join(res_dir, 'search.csv')
    results.to_csv(res_file)

    # best results
    best = fit.get_best_results(results)
    best_file = os.path.join(res_dir, 'fit.csv')
    best.to_csv(best_file)

    # simulate data based on best parameters
    subj_param = best.T.to_dict()
    study_data = data.loc[(data['trial_type'] == 'study')]
    sim = model.generate(
        study_data, {}, subj_param=subj_param, param_def=param_def,
        patterns=patterns, n_rep=n_sim_reps
    )
    sim_file = os.path.join(res_dir, 'sim.csv')
    sim.to_csv(sim_file, index=False)


def split_arg(arg):
    """Split a dash-separated argument."""
    if arg is not None:
        if isinstance(arg, str):
            if arg != 'none':
                split = arg.split('-')
            else:
                split = None
        else:
            split = arg.split('-')
    else:
        split = None
    return split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('patterns_file')
    parser.add_argument('fcf_features')
    parser.add_argument('ff_features')
    parser.add_argument('res_dir')
    parser.add_argument('--sublayers', '-s', action='store_true')
    parser.add_argument('--sublayer-param', '-p', default=None)
    parser.add_argument('--fixed', '-f', default=None)
    parser.add_argument('--n-reps', '-n', type=int, default=1)
    parser.add_argument('--n-jobs', '-j', type=int, default=1)
    parser.add_argument('--tol', '-t', type=float, default=0.00001)
    parser.add_argument('--n-sim-reps', '-r', type=int, default=1)
    parser.add_argument('--include', '-i', default=None)
    args = parser.parse_args()

    fcf = split_arg(args.fcf_features)
    ff = split_arg(args.ff_features)
    include_subjects = split_arg(args.include)
    subpar = split_arg(args.sublayer_param)
    fixed = split_arg(args.fixed)

    main(args.data_file, args.patterns_file, fcf, ff, args.sublayers,
         args.res_dir, subpar, fixed, args.n_reps, args.n_jobs, args.tol,
         args.n_sim_reps, include_subjects)
