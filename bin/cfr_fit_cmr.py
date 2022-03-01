#!/usr/bin/env python
#
# Fit CMR to CFR data.

import os
import argparse
import logging
import pandas as pd
from cymr import cmr
from cymr import network
from cymr import fit
from cfr import framework


def main(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    sublayers,
    res_dir,
    sublayer_param=None,
    fixed_param=None,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    n_sim_reps=1,
    include=None,
):

    os.makedirs(res_dir, exist_ok=True)
    log_file = os.path.join(res_dir, 'log_fit.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )

    # prepare model for search
    logging.info(f'Loading data from {data_file}.')
    data = pd.read_csv(data_file)
    if include is not None:
        data = data.loc[data['subject'].isin(include)]

    # set parameter definitions based on model framework
    model = cmr.CMR()
    param_def = framework.model_variant(
        fcf_features, ff_features, sublayers=sublayers, sublayer_param=sublayer_param
    )
    logging.info(f'Loading network patterns from {patterns_file}.')
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
    json_file = os.path.join(res_dir, 'parameters.json')
    logging.info(f'Saving parameter definition to {json_file}.')
    param_def.to_json(json_file)

    # run individual subject fits
    n = data['subject'].nunique()
    logging.info(
        f'Running {n_reps} parameter optimization repeat(s) for {n} participant(s).'
    )
    logging.info(f'Using {n_jobs} core(s).')
    results = model.fit_indiv(
        data,
        param_def,
        patterns=patterns,
        n_jobs=n_jobs,
        method='de',
        n_rep=n_reps,
        tol=tol,
    )

    # full search information
    res_file = os.path.join(res_dir, 'search.csv')
    logging.info(f'Saving full search results to {res_file}.')
    results.to_csv(res_file)

    # best results
    best = fit.get_best_results(results)
    best_file = os.path.join(res_dir, 'fit.csv')
    logging.info(f'Saving best fitting results to {best_file}.')
    best.to_csv(best_file)

    # simulate data based on best parameters
    subj_param = best.T.to_dict()
    study_data = data.loc[(data['trial_type'] == 'study')]
    logging.info(
        f'Simulating {n_sim_reps} replication(s) with best-fitting parameters.'
    )
    sim = model.generate(
        study_data,
        {},
        subj_param=subj_param,
        param_def=param_def,
        patterns=patterns,
        n_rep=n_sim_reps,
    )
    sim_file = os.path.join(res_dir, 'sim.csv')
    logging.info(f'Saving simulated data to {sim_file}.')
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
    parser = argparse.ArgumentParser(
        description='Run a parameter search to fit a model, then simulate data.'
    )
    parser.add_argument('data_file', help='path to Psifr csv data file')
    parser.add_argument('patterns_file', help='path to network patterns')
    parser.add_argument(
        'fcf_features', help='dash-separated list of item-context associations'
    )
    parser.add_argument(
        'ff_features', help='dash-separated list of item-item associations (or none)'
    )
    parser.add_argument('res_dir', help='directory to save results')
    parser.add_argument(
        '--sublayers',
        '-s',
        action='store_true',
        help='include a sublayer for each feature (default: multiple segments)',
    )
    parser.add_argument(
        '--sublayer-param',
        '-p',
        default=None,
        help='parameters free to vary between sublayers (e.g., B_enc-B_rec)',
    )
    parser.add_argument(
        '--fixed',
        '-f',
        default=None,
        help='dash-separated list of values for fixed parameters (e.g., B_enc_cat=1)',
    )
    parser.add_argument(
        '--n-reps',
        '-n',
        type=int,
        default=1,
        help='number of times to replicate the search',
    )
    parser.add_argument(
        '--n-jobs', '-j', type=int, default=1, help='number of parallel jobs to use'
    )
    parser.add_argument(
        '--tol', '-t', type=float, default=0.00001, help='search tolerance'
    )
    parser.add_argument(
        '--n-sim-reps',
        '-r',
        type=int,
        default=1,
        help='number of experiment replications to simulate',
    )
    parser.add_argument(
        '--include',
        '-i',
        default=None,
        help='dash-separated list of subject to include (default: all in data file)',
    )
    args = parser.parse_args()

    fcf = split_arg(args.fcf_features)
    ff = split_arg(args.ff_features)
    if args.include is not None:
        include_subjects = [int(s) for s in split_arg(args.include)]
    else:
        include_subjects = None
    subpar = split_arg(args.sublayer_param)
    fixed = split_arg(args.fixed)

    main(
        args.data_file,
        args.patterns_file,
        fcf,
        ff,
        args.sublayers,
        args.res_dir,
        subpar,
        fixed,
        args.n_reps,
        args.n_jobs,
        args.tol,
        args.n_sim_reps,
        include_subjects,
    )
