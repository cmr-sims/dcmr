"""Command-line interfaces to work with model variants."""

import os
import shutil
from pathlib import Path
import logging
import functools
import click
import pandas as pd
from psifr import fr
from cymr import cmr
from dcmr import framework
from dcmr import task
from dcmr import reports


def split_arg(arg, sep='-'):
    """Split a dash-separated argument."""
    if arg is not None:
        if isinstance(arg, str):
            if arg != 'none':
                split = arg.split(sep)
            else:
                split = None
        else:
            split = arg.split(sep)
    else:
        split = None
    return split


def process_param_args(
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
):
    sublayer_param = split_arg(sublayer_param)

    fixed_param_list = split_arg(fixed_param)
    fixed_param = {}
    if fixed_param_list is not None:
        for expr in fixed_param_list:
            param_name, val = expr.split('=')
            fixed_param[param_name] = float(val)

    free_param_list = split_arg(free_param)
    free_param = {}
    if free_param_list is not None:
        for expr in free_param_list:
            param_name, val = expr.split('=')
            low, high = val.split(':')
            free_param[param_name] = (float(low), float(high))

    dependent_param_list = split_arg(dependent_param, '$')
    dependent_param = {}
    if dependent_param_list is not None:
        for expr in dependent_param_list:
            param_name, val = expr.split('=')
            dependent_param[param_name] = val

    return sublayer_param, fixed_param, free_param, dependent_param


def configure_model(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
    include,
):
    """Configure a model based on commandline input."""
    # unpack lists
    fcf_features = split_arg(fcf_features)
    ff_features = split_arg(ff_features)
    if include is not None:
        include = [int(s) for s in split_arg(include)]
    sublayer_param, fixed_param, free_param, dependent_param = process_param_args(
        sublayer_param, fixed_param, free_param, dependent_param
    )

    # load data to simulate
    logging.info(f'Loading data from {data_file}.')
    data = pd.read_csv(data_file)
    if include is not None:
        data = data.loc[data['subject'].isin(include)]

    # set parameter definitions based on model framework
    param_def = framework.model_variant(
        fcf_features,
        ff_features,
        sublayers=sublayers,
        scaling=scaling,
        sublayer_param=sublayer_param,
        intercept=intercept,
        fixed_param=fixed_param,
        free_param=free_param,
        dependent_param=dependent_param,
    )
    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    # make sure item index is defined for looking up weight patterns
    if 'item_index' not in data.columns:
        data['item_index'] = fr.pool_index(data['item'], patterns['items'])
        study = fr.filter_data(data, trial_type='study')
        if study['item_index'].isna().any():
            raise ValueError('Patterns not found for one or more items.')
    return data, param_def, patterns


def filter_options(f):
    """Set options for data filtering."""
    @click.option(
        "--include",
        "-i",
        help="dash-separated list of subject to include (default: all in data file)",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def data_options(f):
    """Set data column options."""
    @click.option(
        "--study-keys", 
        '-s', 
        multiple=True, 
        help="names of data columns to include during study",
    )
    @click.option(
        "--recall-keys", 
        '-l', 
        multiple=True, 
        help="names of data columns to include during recall",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def model_options(f):
    """Set options for model configuration."""
    @click.option("--intercept/--no-intercept", default=False)
    @click.option("--sublayers/--no-sublayers", default=False)
    @click.option(
        "--scaling/--no-scaling", 
        default=True,
        help="Include scaling parameters",
    )
    @click.option(
        "--sublayer-param",
        "-p",
        help="parameters free to vary between sublayers (e.g., B_enc-B_rec)",
    )
    @click.option(
        "--fixed-param",
        "-f",
        help="dash-separated list of values for fixed parameters (e.g., B_enc_cat=1)",
    )
    @click.option(
        "--free-param",
        "-e",
        help="dash-separated list of values for free parameter ranges (e.g., B_enc_cat=0:0.8)",
    )
    @click.option(
        "--dependent-param",
        "-a",
        help="list of values for dependent parameter expressions, separated by $ (e.g., B_enc_cat=B_enc_use$wr_cat=1 - w0)",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def fit_options(f):
    @click.option(
        "--n-reps",
        "-n",
        type=int,
        default=1,
        help="number of times to replicate the search",
    )
    @click.option(
        "--n-jobs", "-j", type=int, default=1, help="number of parallel jobs to use"
    )
    @click.option("--tol", "-t", type=float, default=0.00001, help="search tolerance")
    @click.option("--init", default='latinhypercube', help="search initialization method")
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def sim_options(f):
    @click.option(
        "--n-sim-reps",
        "-r",
        type=int,
        default=1,
        help="number of experiment replications to simulate",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def report_options(f):
    @click.option(
        "--category/--no-category", default=False, help="include category analyses"
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


def xval_options(f):
    @click.option(
        "--n-folds",
        "-d",
        type=int,
        help="number of cross-validation folds to run",
    )
    @click.option(
        "--fold-key",
        "-k",
        help="events column to use when defining cross-validation folds",
    )
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)
    return wrapper


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fcf_features")
@click.argument("ff_features")
@click.argument("res_dir", type=click.Path())
@model_options
@fit_options
@sim_options
@filter_options
@data_options
def fit_cmr(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    res_dir,
    intercept,
    sublayers,
    scaling,
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
    n_reps,
    n_jobs,
    tol,
    init,
    n_sim_reps,
    include,
    study_keys,
    recall_keys,
):
    """
    Run a parameter search to fit a model and simulate data.

    Fit the data in DATA_FILE, setting model weights using the patterns
    in PATTERNS_FILE, with Mfc and Mcf sublayers FCF_FEATURES and Mff
    weights FF_FEATURES, and saving results to RES_DIR. Features may
    include multiple features, separated by dashes.
    """
    os.makedirs(res_dir, exist_ok=True)
    log_file = os.path.join(res_dir, 'log_fit.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )

    # set up data and model based on script input
    data, param_def, patterns = framework.configure_model(
        data_file,
        patterns_file,
        fcf_features,
        ff_features,
        intercept,
        sublayers,
        scaling,
        sublayer_param,
        fixed_param,
        free_param,
        dependent_param,
        include,
    )

    # fit parameters, simulate using fitted parameters, and save results
    framework.run_fit(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_jobs, 
        n_reps, 
        tol, 
        init, 
        n_sim_reps,
        study_keys,
        recall_keys,
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("res_dir", type=click.Path())
@click.option("--disrupt/--no-disrupt", default=True)
@fit_options
@sim_options
@filter_options
def fit_cmr_cfr_disrupt(
    data_file,
    patterns_file,
    res_dir,
    disrupt=True,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    init='latinhypercube',
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

    logging.info(f'Loading data from {data_file}.')
    if include is not None:
        include = include.split('-')
    data = task.read_study_recall(data_file, include=include)

    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    if disrupt:
        param_def = framework.model_variant(
            ['loc', 'cat', 'use'], 
            sublayers=True,
            free_param={
                'T': (0, 1),
                'B_disrupt': (0, 1),
                'B_enc': (0.1, 1),
            },
            sublayer_param=[
                'B_enc', 
                'B_rec', 
                'Lfc', 
                'Lcf',
                'B_disrupt',
                'B_distract',
            ],
            fixed_param={
                'B_rec_cat': 1, 
                'B_rec_use': 1, 
                'B_disrupt_loc': 0, 
                'B_disrupt_use': 0, 
                'B_disrupt_cat': 0.9,
                'B_retention': 0, 
                'B_start': 0,
            },
            dynamic_param={
                ('study', 'trial'): {
                    'B_distract_loc': 'where((block != 1) & (block_pos == 1), B_disrupt_loc, 0)',
                    'B_distract_cat': 'where((block != 1) & (block_pos == 1), B_disrupt_cat, 0)',
                    'B_distract_use': 'where((block != 1) & (block_pos == 1), B_disrupt_use, 0)',
                }
            },
            intercept=False,
            list_context=True,
            distraction=True,
        )
        param_def.set_free(
            {
                'Lfc_cat_raw': (0.1, 0.9),
                'Lcf_cat_raw': (0.1, 0.9),
                'Lfc_use_raw': (0.1, 0.9),
                'Lcf_use_raw': (0.1, 0.9),
            }
        )
    else:
        param_def = framework.model_variant(
            ['loc', 'cat', 'use'], 
            sublayers=True,
            free_param={
                'T': (0, 1),
                'B_enc': (0, 1),
            },
            sublayer_param=[
                'B_enc', 
                'B_rec', 
                'Lfc', 
                'Lcf',
            ],
            fixed_param={
                'B_rec_cat': 1, 
                'B_rec_use': 1, 
                'B_start': 0,
                'Lfc_cat_raw': 0.5,
                'Lcf_cat_raw': 0.5,
                'Lfc_use_raw': 0.5,
                'Lcf_use_raw': 0.5,
            },
            intercept=False,
            list_context=True,
        )
    del param_def.fixed['T']
    param_def.set_free(w0=(0.1, 1.9), w1=(0.1, 1))
    param_def.set_dependent(wr_cat="2 - w0")

    # fit parameters, simulate using fitted parameters, and save results
    study_keys = ['block', 'block_pos']
    framework.run_fit(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_jobs, 
        n_reps, 
        tol, 
        init,
        n_sim_reps, 
        study_keys,
    )

    # evaluate using cross-validation
    n_folds = None
    fold_key = "session"
    framework.run_xval(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_folds, 
        fold_key, 
        n_reps, 
        n_jobs, 
        tol,
        init,
        study_keys,
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("res_dir", type=click.Path())
@fit_options
@sim_options
@filter_options
def fit_cmr_cdcatfr2(
    data_file,
    patterns_file,
    res_dir,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    init='latinhypercube',
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

    logging.info(f'Loading data from {data_file}.')
    if include is not None:
        include = include.split('-')
    data = task.read_study_recall(data_file, include=include)

    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    param_def = framework.model_variant(
        ['loc', 'cat', 'use'], 
        sublayers=True,
        free_param={
            'T': (0, 1),
            'B_enc': (0, 1),
            'B_distract_raw': (0, 0.4), 
            'B_disrupt': (0, 1),
            'X10': (0, 1),
            'X11': (0, 1),
            'X12': (0, 1),
            'X20': (0, 1),
            'X21': (0, 1),
            'X22': (0, 1),
        },
        sublayer_param=[
            'B_enc', 
            'B_rec', 
            'Lfc', 
            'Lcf', 
            'B_distract', 
            'B_retention', 
            'B_distract_raw', 
            'B_retention_raw',
        ],
        fixed_param={'B_rec_cat': 1, 'B_rec_use': 1},
        dependent_param={
            'B_retention_raw_loc': 'B_distract_raw_loc',
            'B_retention_raw_cat': 'B_distract_raw_cat',
            'B_retention_raw_use': 'B_distract_raw_use',
        },
        dynamic_param={
            ('study', 'list'): {
                'B_distract_loc': 'clip(B_distract_raw_loc * distractor, 0, 1)',
                'B_distract_use': 'clip(B_distract_raw_use * distractor, 0, 1)',
                'B_retention_loc': 'clip(B_retention_raw_loc * distractor, 0, 1)',
                'B_retention_cat': 'clip(B_retention_raw_cat * distractor, 0, 1)',
                'B_retention_use': 'clip(B_retention_raw_use * distractor, 0, 1)',
                'X1': 'where(distractor == 0, X10, where(distractor == 2.5, X11, X12))',
                'X2': 'where(distractor == 0, X20, where(distractor == 2.5, X21, X22))',
            },
            ('study', 'trial'): {
                'B_distract_cat': 'clip(B_distract_raw_cat * distractor + where((block != 1) & (block_pos == 1), B_disrupt, 0), 0, 1)',
            }
        },
        intercept=False,
        list_context=True,
        distraction=True,
    )
    del param_def.fixed['T']
    del param_def.free['X1']
    del param_def.free['X2']
    param_def.set_free(w0=(0, 2))
    param_def.set_dependent(wr_cat="2 - w0")

    # fit parameters, simulate using fitted parameters, and save results
    study_keys = ['distractor', 'block', 'block_pos']
    fit = framework.run_fit(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_jobs, 
        n_reps, 
        tol, 
        init,
        n_sim_reps, 
        study_keys,
    )
    subj_param = best.T.to_dict()

    # plot results by condition
    sim = pd.read_csv(os.path.join(res_dir, 'sim.csv'))
    distract_list = [0.0, 2.5, 7.5]
    for distract in distract_list:
        report_name = f'distract{distract}'
        data_filter = f'distractor == {distract}'
        reports.plot_fit(
            data, 
            sim, 
            {},
            subj_param,
            param_def,
            patterns, 
            fit,
            res_dir, 
            report_name=report_name, 
            ext='svg', 
            study_keys=study_keys,
            category=True,
            data_filter=data_filter,
        )

    # evaluate using cross-validation
    n_folds = None
    fold_key = "session"
    framework.run_xval(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_folds, 
        fold_key, 
        n_reps, 
        n_jobs, 
        tol,
        init,
        study_keys,
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("res_dir", type=click.Path())
@fit_options
@sim_options
@filter_options
def fit_cmr_asymfr(
    data_file,
    patterns_file,
    res_dir,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    init='latinhypercube',
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

    logging.info(f'Loading data from {data_file}.')
    if include is not None:
        include = include.split('-')
    data = task.read_study_recall(
        data_file, block=False, block_category=False, include=include
    )

    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    param_def = framework.model_variant(
        ['loc', 'use'], 
        sublayers=True,
        intercept=True,
        sublayer_param=[
            'B_enc', 
            'B_rec', 
            'Lfc', 
            'Lcf',
        ],
        free_param={
            'Aff': (-2, 2),
            'w00': (0, 1),
            'w01': (0, 1),
            'w02': (0, 1),
            'X10': (0, 1),
            'X11': (0, 1),
            'X12': (0, 1),
            'X20': (0, 2),
            'X21': (0, 2),
            'X22': (0, 2),
        },
        fixed_param={'B_rec_use': 1, 'w0': 1},
        dynamic_param={
            ('study', 'list'): {
                'w0': 'where(list_type == "same", w00, where(list_type == "mixed", w01, w02))',
                'X1': 'where(list_type == "same", X10, where(list_type == "mixed", X11, X12))',
                'X2': 'where(list_type == "same", X20, where(list_type == "mixed", X21, X22))',
            }
        }
    )
    del param_def.free['X1']
    del param_def.free['X2']

    # fit parameters, simulate using fitted parameters, and save results
    framework.run_fit(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_jobs, 
        n_reps, 
        tol, 
        init,
        n_sim_reps, 
        study_keys=['list_type'],
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("res_dir", type=click.Path())
@fit_options
@sim_options
@filter_options
def fit_cmr_incidental(
    data_file,
    patterns_file,
    res_dir,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    init='latinhypercube',
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

    logging.info(f'Loading data from {data_file}.')
    if include is not None:
        include = include.split('-')
    data = task.read_study_recall(
        data_file, block=False, block_category=False, include=include
    )

    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    param_def = framework.model_variant(
        ['loc', 'use'], 
        sublayers=True,
        free_param={
            'B_distract_raw': (0, 1), 
            'w0_intent': (0, 1),
            'w0_incid': (0, 1),
            'X1_intent': (0, 1),
            'X1_incid': (0, 1),
            'X2_intent': (0, 5),
            'X2_incid': (0, 5),
        },
        sublayer_param=[
            'B_enc', 
            'B_rec', 
            'B_distract', 
            'B_retention', 
            'B_distract_raw', 
            'B_retention_raw',
        ],
        fixed_param={'B_rec_use': 1, 'w0': 0.5},
        dependent_param={
            'B_retention_raw_loc': 'B_distract_raw_loc',
            'B_retention_raw_use': 'B_distract_raw_use',
        },
        dynamic_param={
            ('study', 'list'): {
                'B_distract_loc': 'where(distractor == 16, B_distract_raw_loc, 0)',
                'B_distract_use': 'where(distractor == 16, B_distract_raw_use, 0)',
                'B_retention_loc': 'where(retention == 16, B_retention_raw_loc, 0)',
                'B_retention_use': 'where(retention == 16, B_retention_raw_use, 0)',
                'w0': 'where(encoding == "intentional", w0_intent, w0_incid)',
                'X1': 'where(encoding == "intentional", X1_intent, X1_incid)',
                'X2': 'where(encoding == "intentional", X2_intent, X2_incid)',
            }
        }
    )
    del param_def.free['X1']
    del param_def.free['X2']
    param_def.set_options(distraction=True)

    # fit parameters, simulate using fitted parameters, and save results
    framework.run_fit(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_jobs, 
        n_reps, 
        tol, 
        init,
        n_sim_reps, 
        study_keys=['distractor', 'retention', 'encoding'],
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fcf_features")
@click.argument("ff_features")
@click.argument("res_dir", type=click.Path())
@model_options
@xval_options
@fit_options
@filter_options
@data_options
def xval_cmr(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    res_dir,
    intercept,
    sublayers,
    scaling,
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
    n_folds,
    fold_key,
    n_reps,
    n_jobs,
    tol,
    init,
    include,
    study_keys,
    recall_keys,
):
    """
    Evaluate a model using cross-validation.
    
    Evaluate data in DATA_FILE, setting model weights using the 
    patterns in PATTERNS_FILE, with Mfc and Mcf sublayers FCF_FEATURES 
    and Mff weights FF_FEATURES, and saving results to RES_DIR. 
    Features may include multiple features, separated by dashes.
    """
    os.makedirs(res_dir, exist_ok=True)
    log_file = os.path.join(res_dir, 'log_xval.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )

    # set up data and model based on script input
    data, param_def, patterns = framework.configure_model(
        data_file,
        patterns_file,
        fcf_features,
        ff_features,
        intercept,
        sublayers,
        scaling,
        sublayer_param,
        fixed_param,
        free_param,
        dependent_param,
        include,
    )

    # split data into folds, fit to training set, evaluate on testing set
    framework.run_xval(
        res_dir, 
        data, 
        param_def, 
        patterns, 
        n_folds, 
        fold_key, 
        n_reps, 
        n_jobs, 
        tol, 
        init,
        study_keys,
        recall_keys,
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fit_dir", type=click.Path(exists=True))
@click.argument("report_name")
@click.option(
    "--fixed-param",
    "-f",
    help="dash-separated list of values for fixed parameters (e.g., B_enc_cat=1)",
)
@click.option(
    "--dependent-param",
    "-a",
    help="dash-separated list of values for dependent parameter expressions (e.g., B_enc_cat=B_enc_use)",
)
@sim_options
@click.option("--study-keys", '-k', multiple=True)
@report_options
def adjust_sim(
    data_file, 
    patterns_file, 
    fit_dir, 
    report_name, 
    fixed_param, 
    dependent_param, 
    n_sim_reps, 
    study_keys,
    category,
):
    """
    Run a simulation by adjusting an existing fit.

    Given data in DATA_FILE, model weights in the PATTERNS_FILE, and
    fit results in FIT_DIR, run an adjusted simulation with changes to
    parameters and save results to a report subdirectory called 
    REPORT_NAME.
    """
    # load trials to simulate
    data = task.read_study_recall(data_file)
    study_data = data.loc[(data['trial_type'] == 'study')]

    # get model, patterns, and weights
    model = cmr.CMR()
    patterns = cmr.load_patterns(patterns_file)
    param_file = os.path.join(fit_dir, 'parameters.json')
    param_def = cmr.read_config(param_file)

    # load parameters
    fit_file = os.path.join(fit_dir, 'fit.csv')
    subj_param = framework.read_fit_param(fit_file)
    fit = pd.read_csv(fit_file)

    # update parameters
    sublayer_param, fixed_param, free_param, dependent_param = process_param_args(
        None,
        fixed_param, 
        None,
        dependent_param,
    )
    if fixed_param:
        group_param = fixed_param
        for par, val in group_param.items():
            for subj, param in subj_param.items():
                if par in param:
                    del subj_param[subj][par]
    else:
        group_param = None

    if dependent_param:
        param_def.set_dependent(dependent_param)

    # run simulation
    sim = model.generate(
        study_data, 
        group_param, 
        subj_param, 
        param_def, 
        patterns, 
        n_rep=n_sim_reps, 
        study_keys=list(study_keys),
    )
    report_dir = os.path.join(fit_dir, report_name)
    os.makedirs(report_dir, exist_ok=True)
    sim.to_csv(os.path.join(report_dir, 'sim.csv'))

    # make a report of the fit
    reports.plot_fit(
        data, 
        sim, 
        group_param,
        subj_param,
        param_def,
        patterns, 
        fit,
        report_dir, 
        study_keys=list(study_keys), 
        category=category,
    )


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fit_dir", type=click.Path(exists=True))
@click.option("--n-rep", "-r", type=int, default=1)
@click.option("--study-key", multiple=True)
def sim_cmr(data_file, patterns_file, fit_dir, n_rep=1, study_key=None):
    """Run a simulation using best-fitting parameters."""
    # load trials to simulate
    data = task.read_study_recall(data_file)
    study_data = data.loc[(data['trial_type'] == 'study')]

    # get model, patterns, and weights
    model = cmr.CMR()
    patterns = cmr.load_patterns(patterns_file)
    param_file = os.path.join(fit_dir, 'parameters.json')
    param_def = cmr.read_config(param_file)

    # load parameters
    fit_file = os.path.join(fit_dir, 'fit.csv')
    subj_param = framework.read_fit_param(fit_file)

    # run simulation
    sim = model.generate(
        study_data, 
        {}, 
        subj_param, 
        param_def, 
        patterns, 
        n_rep=n_rep, 
        study_keys=list(study_key),
    )

    # save
    sim_file = os.path.join(fit_dir, 'sim.csv')
    sim.to_csv(sim_file, index=False)


@click.command()
@click.argument("out_dir", type=click.Path())
@click.argument("split_dirs", nargs=-1, type=click.Path(exists=True))
def join_xval(out_dir, split_dirs):
    """Join a split-up cross-validation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    param_file = out_dir / "parameters.json"
    search_file = out_dir / "xval_search.csv"
    xval_file = out_dir / "xval.csv"
    log_file = out_dir / "log_xval.txt"
    log = ""
    search_list = []
    xval_list = []
    for split_dir in split_dirs:
        print(f"Loading data from {split_dir}.")
        split_dir = Path(split_dir)

        # copy parameters file
        if not param_file.exists():
            split_param_file = split_dir / "parameters.json"
            if split_param_file.exists():
                shutil.copy(split_param_file, param_file)
            else:
                raise IOError(f"Parameters file does not exist: {split_param_file}")

        # add to log
        split_log_file = split_dir / "log_xval.txt"
        if not split_log_file.exists():
            raise IOError(f"Log file does not exist: {split_log_file}")
        log += split_log_file.read_text()

        # add to search
        split_search_file = split_dir / "xval_search.csv"
        if not split_search_file.exists():
            raise IOError(f"Search file does not exist: {split_search_file}")
        split_search = pd.read_csv(split_search_file)
        search_list.append(split_search)

        # add to xval
        split_xval_file = split_dir / "xval.csv"
        if not split_xval_file.exists():
            raise IOError(f"X-val file does not exist: {split_xval_file}")
        split_xval = pd.read_csv(split_xval_file)
        xval_list.append(split_xval)

    # write out concatenated files
    print(f"Writing data to {out_dir}.")
    log_file.write_text(log)
    search = pd.concat(search_list, ignore_index=True)
    search.sort_values(["fold", "subject", "rep"]).to_csv(search_file, index=False)
    xval = pd.concat(xval_list, ignore_index=True)
    xval.sort_values(["fold", "subject"]).to_csv(xval_file, index=False)


@click.command()
@click.argument("data_file")
@click.argument("patterns_file")
@click.argument("fit_dir")
@click.option("--data-filter", "-d", help="filter to apply to data before plotting")
@click.option("--report-name", "-r", help="name of the report directory")
@click.option("--ext", "-e", default="svg", help="figure file type (default: svg)")
@click.option("--study-keys", "-k", multiple=True, help="study keys for simulation")
@report_options
def run_plot_fit(
    data_file, 
    patterns_file, 
    fit_dir, 
    data_filter, 
    report_name, 
    ext, 
    study_keys,
    category,
):
    """
    Create an HTML report with fit diagnostics.

    Given fitted data in DATA_FILE, model weights specified in
    PATTERNS_FILE, and fit results in FIT_DIR, create a report with
    plots of common summary statistics, fitted parameters, and model
    diagnostics.

    Optionally, statistics may take stimulus category into account if 
    the category option is set and there is a column in DATA_FILE 
    called "category". These statistics include category clustering and
    versions of other organization statistics that are split by whether
    transitions are within- or across-category.
    """
    log_file = os.path.join(fit_dir, 'log_plot.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )
    logging.info(f'Plotting fitted simulation data in {fit_dir}.')

    # load data and simulated data
    sim_file = os.path.join(fit_dir, 'sim.csv')
    logging.info(f'Loading data from {data_file}.')
    data = task.read_study_recall(data_file, block=True, block_category=False)
    logging.info(f'Loading simulation from {sim_file}.')
    sim = task.read_study_recall(sim_file, block=True, block_category=False)

    # load patterns
    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)

    logging.info(f'Creating {report_name} report.')
    if not study_keys:
        study_keys = None
    else:
        study_keys = list(study_keys)

    # prep simulation
    param_def = cmr.read_config(os.path.join(fit_dir, 'parameters.json'))
    fit_file = os.path.join(fit_dir, 'fit.csv')
    subj_param = framework.read_fit_param(fit_file)
    fit = pd.read_csv(fit_file)

    # generate plots and report
    if report_name is not None:
        report_dir = os.path.join(fit_dir, report_name)
    else:
        report_dir = fit_dir
    reports.plot_fit(
        data, 
        sim, 
        {},
        subj_param,
        param_def,
        patterns, 
        fit,
        report_dir, 
        ext, 
        study_keys, 
        category, 
        data_filter,
    )
