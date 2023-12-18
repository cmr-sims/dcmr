"""Fit and simulate data using CMR."""

import os
from pathlib import Path
import json
import logging
from itertools import combinations
from pkg_resources import resource_filename
import numpy as np
import pandas as pd
import click
from cymr import cmr
from cymr import fit
from psifr import fr
from cymr.cmr import CMRParameters
from cfr import task


class WeightParameters(CMRParameters):
    """
    Manage CFR parameters.

    Model Parameters
    ----------------
    w_loc : float
        Relative weight to apply to localist patterns. [0, Inf]

    w_cat : float
        Relative weight to apply to category patterns. [0, Inf]

    w_sem : float
        Relative weight to apply to word2vec patterns. [0, Inf]

    Lfc : float
        Learning rate of item-context weights. [0, 1]

    Lcf : float
        Learning rate of context-item weights. [0, 1]

    P1 : float
        Additional context-item learning for first item. [0, Inf]

    P2 : float
        Decay rate for primacy learning rate gradient. [0, Inf]

    B_enc : float
        Integration rate during encoding. [0, 1]

    B_start : float
        Integration rate of start context reinstatement. [0, 1]

    B_rec : float
        Integration rate during recall. [0, 1]

    X1 : float
        Probability of not recalling any items. [0, 1]

    X2 : float
        Shape parameter of exponential function increasing stop
        probability by output position. [0, Inf]
    """

    def set_scaling_param(self, scaling_type, weights, upper=1):
        """
        Add scaling parameters for patterns or similarity.

        Parameters
        ----------
        scaling_type : {'similarity', 'vector'}
            Type of matrix to be scaled. This determines how weights
            are normalized.

        weights : list of str
            Labels of weights to include.

        upper : float
            Upper bound of parameters beyond the first two.
        """
        if scaling_type == 'vector':
            prefix = 'w'
            ssw = ' + '.join([f'wr_{name}**2' for name in weights])
            denom = f'sqrt({ssw})'
        elif scaling_type == 'similarity':
            prefix = 's'
            sw = ' + '.join([f'sr_{name}' for name in weights])
            denom = f'({sw})'
        else:
            raise ValueError(f'Invalid scaling type: {scaling_type}')

        n_weight = len(weights)
        w_param = [f'{prefix}{n}' for n in range(n_weight - 1)]

        n = 0
        m = 0
        rescaled = {}
        scaling_param = {}
        for name in weights:
            param = f'{prefix}_{name}'
            raw_param = f'{prefix}r_{name}'

            if n_weight == 1:
                # if only one, no weight parameter necessary
                scaling_param[name] = None
                continue

            # set up scaling parameter and translate to original name
            if n == 0:
                ref_param = w_param[m]
                self.set_free({ref_param: (0, 1)})
                self.set_dependent({raw_param: ref_param})
                m += 1
            elif n == 1:
                self.set_dependent({raw_param: f'1 - {ref_param}'})
            else:
                new_param = w_param[m]
                self.set_free({new_param: (0, upper)})
                self.set_dependent({raw_param: new_param})
                m += 1

            # set up rescaling
            rescaled.update({param: f'{raw_param} / {denom}'})
            n += 1
            scaling_param[name] = param
        self.set_dependent(rescaled)
        return scaling_param

    def set_intercept_param(self, connects, lower, upper):
        """Set intercept parameters."""
        intercept_param = {}
        for connect in connects:
            new_param = f'A{connect}'
            intercept_param[connect] = new_param
            self.set_free({new_param: (lower, upper)})
        return intercept_param

    def set_region_weights(self, connect, scaling_param, pre_param):
        """Sets weights within sublayer regions."""
        for weight, scaling in scaling_param.items():
            if scaling is not None:
                expr = f'{pre_param} * {scaling} * {weight}'
            else:
                expr = f'{pre_param} * {weight}'
            self.set_weights(connect, {(('task', 'item'), ('task', weight)): expr})

    def set_sublayer_weights(
        self, connect, scaling_param, pre_param, intercept_param=None
    ):
        """Set weights for different sublayers."""
        for weight, scaling in scaling_param.items():
            # get pre weighting parameter
            if isinstance(pre_param, str):
                pre = pre_param
            else:
                pre = pre_param[weight]

            # get intercept parameter, if any
            if intercept_param is None:
                intercept = None
            elif isinstance(intercept_param, str):
                intercept = intercept_param
            else:
                intercept = intercept_param[weight]

            # set weights expression
            if scaling is not None:
                if intercept is not None:
                    expr = f'{intercept} + {pre} * {scaling} * {weight}'
                else:
                    expr = f'{pre} * {scaling} * {weight}'
            else:
                if intercept is not None:
                    expr = f'{intercept} + {pre} * {weight}'
                else:
                    expr = f'{pre} * {weight}'
            self.set_weights(connect, {(('task', 'item'), (weight, 'item')): expr})

    def set_item_weights(self, scaling_param, pre_param, intercept_param=None):
        """Set item-item weights."""
        weight_expr = []
        for weight, scaling in scaling_param.items():
            if scaling is not None:
                expr = f'{scaling} * {weight}'
            else:
                expr = weight
            weight_expr.append(expr)
        expr = ' + '.join(weight_expr)
        if intercept_param is None:
            w_expr = f'{pre_param} * ({expr})'
        else:
            w_expr = f'{intercept_param["ff"]} + {pre_param} * ({expr})'
        self.set_weights('ff', {('task', 'item'): w_expr})

    def set_learning_sublayer_param(self, L_name, D_name):
        """Set dependent sublayer parameters for learning."""
        # free up the learning rate
        self.set_free_sublayer_param([L_name], '_raw')

        # remove existing generic definition
        if D_name in self.dependent:
            del self.dependent[D_name]

        L_param = {}
        D_param = {}
        for weight in self.sublayers['c']:
            source = f'{L_name}_{weight}_raw'
            dependent = f'{D_name}_{weight}'
            self.set_dependent({dependent: f'1 - {source}'})

            L_param[weight] = source
            D_param[weight] = dependent
        return L_param, D_param

    def set_weight_sublayer_param(self, scaling_param, suffix=None):
        """Set scaling of sublayer learning rates."""
        for weight in self.sublayers['c']:
            scaling = scaling_param[weight]

            # if no scaling, do not need to set learning rates to vary
            if scaling is None:
                continue

            weight_param = {}
            for param in ['Lfc', 'Lcf']:
                sub_param = f'{param}_{weight}'
                if suffix is not None and suffix[param] is not None:
                    raw_param = sub_param + suffix[param]
                    self.set_dependent({sub_param: f'{raw_param} * {scaling}'})
                else:
                    self.set_dependent({sub_param: f'{param} * {scaling}'})
                weight_param[param] = sub_param
            self.set_sublayer_param('c', weight, weight_param)

    def set_free_sublayer_param(self, param_names, suffix=None):
        """Set sublayer parameters to be free."""
        for param in param_names:
            if param not in self.free:
                raise ValueError(f'No range defined for {param}.')

            # make a copy of the base parameter for each sublayer
            for weight in self.sublayers['c']:
                param_name = f'{param}_{weight}'
                if suffix is not None:
                    param_name += suffix
                self.set_free({param_name: self.free[param]})
                self.set_sublayer_param('c', weight, {param: param_name})

            # remove the base parameter from the list of free variables
            if param in self.free:
                del self.free[param]


def model_variant(
    fcf_features,
    ff_features=None,
    sublayers=False,
    sublayer_param=None,
    intercept=False,
    fixed_param=None,
):
    """Define parameters for a model variant."""
    wp = WeightParameters()
    wp.set_fixed(T=0.1)
    wp.set_free(
        Lfc=(0, 1),
        Lcf=(0, 1),
        P1=(0, 10),
        P2=(0.1, 5),
        B_enc=(0, 1),
        B_start=(0, 1),
        B_rec=(0, 1),
        X1=(0, 1),
        X2=(0, 5),
    )
    wp.set_dependent(Dfc='1 - Lfc', Dcf='1 - Lcf')

    if intercept:
        intercept_param = wp.set_intercept_param(['ff'], -1, 1)
    else:
        intercept_param = None

    if fcf_features:
        # set global weight scaling
        scaling_param = wp.set_scaling_param('vector', fcf_features)
        if sublayers:
            wp.set_sublayers(f=['task'], c=fcf_features)

            # set sublayer L and D parameters
            Dfc = 'Dfc'
            Dcf = 'Dcf'
            suffix = {'Lfc': None, 'Lcf': None}
            if sublayer_param is not None:
                if 'Lfc' in sublayer_param:
                    Lfc, Dfc = wp.set_learning_sublayer_param('Lfc', 'Dfc')
                    sublayer_param.remove('Lfc')
                    suffix['Lfc'] = '_raw'
                if 'Lcf' in sublayer_param:
                    Lcf, Dcf = wp.set_learning_sublayer_param('Lcf', 'Dcf')
                    sublayer_param.remove('Lcf')
                    suffix['Lcf'] = '_raw'
                wp.set_free_sublayer_param(sublayer_param)

            # set weights based on fixed or varying D parameters
            wp.set_sublayer_weights('fc', scaling_param, Dfc)
            wp.set_sublayer_weights('cf', scaling_param, Dcf)

            # set learning rate to vary by sublayer
            wp.set_weight_sublayer_param(scaling_param, suffix)
        else:
            # set weights based on fixed D parameters
            wp.set_sublayers(f=['task'], c=['task'])
            wp.set_region_weights('fc', scaling_param, 'Dfc')
            wp.set_region_weights('cf', scaling_param, 'Dcf')

    if ff_features:
        scaling_param = wp.set_scaling_param('similarity', ff_features)
        wp.set_item_weights(scaling_param, 'Dff', intercept_param)
        wp.set_free(Dff=(0, 10))
    elif intercept_param is not None:
        intercept = intercept_param['ff']
        expr = f'{intercept} * ones(loc.shape)'
        wp.set_weights('ff', {('task', 'item'): expr})

    # fix parameters if specified
    if fixed_param is not None:
        for param_name, val in fixed_param.items():
            wp.set_fixed({param_name: val})
            if param_name not in wp.free:
                raise ValueError(f'Parameter {param_name} is not free.')
            del wp.free[param_name]
    return wp


def read_fit_param(fit_file):
    """Read subject parameters from a fit results file."""
    fit = pd.read_csv(fit_file, index_col=0)
    fit = fit.drop(['rep', 'logl', 'n', 'k'], axis=1)
    param = fit.T.to_dict()
    return param


def read_fit_weights(param_file):
    """Read weights from a parameters file."""
    with open(param_file, 'r') as f:
        wp = json.load(f)
    weights = wp['weights']
    return weights


def read_model_spec(def_file):
    """Read model specification file as a series."""
    with open(def_file, 'r') as f:
        model_def = json.load(f)

    value = {**model_def['fixed'], **model_def['free'], **model_def['dependent']}
    kind = {}
    for par in model_def['fixed'].keys():
        kind[par] = 'fixed'
    for par in model_def['free'].keys():
        kind[par] = 'free'
    for par in model_def['dependent'].keys():
        kind[par] = 'dependent'

    df = pd.DataFrame([value, kind], index=['value', 'kind'])
    return df.T


def read_model_specs(fit_dir, models, model_names=None):
    """Read model definitions for multiple models."""
    if model_names is None:
        model_names = models

    spec_list = []
    for model in models:
        spec_file = os.path.join(fit_dir, model, 'parameters.json')
        if not os.path.exists(spec_file):
            spec_file = os.path.join(fit_dir, model, 'xval_parameters.json')
            if not os.path.exists(spec_file):
                raise IOError(f'Parameters file not found: {spec_file}')
        spec = read_model_spec(spec_file)
        spec_list.append(spec)
    model_defs = pd.concat(spec_list, keys=model_names)
    model_defs.index.rename(['model', 'param'], inplace=True)
    return model_defs


def read_model_fits(fit_dir, models, model_names=None, param_map=None):
    """Read fit results for multiple models."""
    if model_names is None:
        model_names = models

    res_list = []
    for model in models:
        fit_file = os.path.join(fit_dir, model, 'fit.csv')
        res_model = pd.read_csv(fit_file)
        res_list.append(res_model)
    res = pd.concat(res_list, axis=0, keys=model_names)
    res = res.reset_index(level=1, drop=True)
    res.index.rename('model', inplace=True)
    res = res.set_index('subject', append=True)

    # map overall parameters to subset parameters
    if param_map is not None:
        for key, params in param_map.items():
            inc = res[key].notna()
            for param in params:
                res.loc[inc, param] = res.loc[inc, key]
        res.drop(columns=list(param_map.keys()), inplace=True)
    return res


def read_model_xvals(fit_dir, models, model_names=None):
    """Read cross-validation results for multiple models."""
    if model_names is None:
        model_names = models

    res_list = []
    for model in models:
        fit_file = os.path.join(fit_dir, model, 'xval.csv')
        res_model = pd.read_csv(fit_file)
        res_list.append(res_model)
    res = pd.concat(res_list, axis=0, keys=model_names)
    res = res.reset_index(level=1, drop=True)
    res.index.rename('model', inplace=True)
    res = res.set_index('subject', append=True)
    return res


def read_model_sims(
    data_file, fit_dir, models, model_names=None, block=False, block_category=False
):
    """Read simulated data for multiple models."""
    if model_names is None:
        model_names = models

    data_list = []
    obs_data = task.read_free_recall(
        data_file, block=block, block_category=block_category
    )
    for model in models:
        sim_file = os.path.join(fit_dir, model, 'sim.csv')
        sim_data = task.read_free_recall(
            sim_file, block=block, block_category=block_category
        )
        data_list.append(sim_data)
    data_list.append(obs_data)
    data = pd.concat(data_list, axis=0, keys=model_names + ['Data'])
    data.index.rename(['source', 'trial'], inplace=True)
    return data


def get_sim_models(study, model_set, included=None):
    """Get a list of models for a study."""
    list_file = resource_filename('cfr', f'models/{study}.json')
    with open(list_file, 'r') as f:
        model_list = json.load(f)
        if included is not None:
            model_dict = {
                s[model_set]: s['full']
                for short_name, s in model_list.items()
                if s[model_set] in included
            }
        else:
            model_dict = {
                s[model_set]: s['full'] for short_name, s in model_list.items()
            }
        model_names = list(model_dict.keys())
        models = list(model_dict.values())
        return models, model_names


def get_sim2_models(study):
    """Get main models used in simulation 2."""
    if study == 'cfr':
        model_dict = {
            'DCMR': 'cmrs_fcf-loc-cat-use',
            'DCMR-Variable': 'cmrs_fcf-loc-cat-use_sl-B_enc-B_rec',
            'DCMR-Restricted': 'cmrs_fcf-loc-cat-use_sl-B_enc-B_rec_fix-B_rec_cat1-B_rec_use1',
            'CMR MP16': 'cmrs_fcf-loc_ff-cat-use',
        }
    elif study == 'peers':
        model_dict = {
            'DCMR': 'cmrs_fcf-loc-use',
            'DCMR-Variable': 'cmrs_fcf-loc-use_sl-B_enc-B_rec',
            'DCMR-Restricted': 'cmrs_fcf-loc-use_sl-B_enc-B_rec_fix-B_rec_use1',
            'DCMR-NoSemDrift': 'cmrs_fcf-loc-use_sl-B_enc-B_rec_fix-B_enc_use1-B_rec_use1',
            'CMR MP16': 'cmrs_fcf-loc_ff-use',
        }
    else:
        raise ValueError(f'Invalid study: {study}')
    models = list(model_dict.values())
    model_names = list(model_dict.keys())
    return models, model_names


def get_sim2_all_models(study):
    """Get all models used in simulation 2."""
    if study == 'cfr':
        fixed, model_names = generate_restricted_models()
        models = [
            f'cmrs_fcf-loc-cat-use_sl-B_enc-B_rec_fix-{f.replace("=", "")}'
            for f in fixed
        ]
        models = (
            ['cmrs_fcf-loc-cat-use', 'cmrs_fcf-loc-cat-use_sl-B_enc-B_rec']
            + models
            + ['cmrs_fcf-loc_ff-cat-use']
        )
        model_names = ['ICD', 'UR'] + model_names + ['MP16']
    elif study == 'peers':
        models = [
            'cmrs_fcf-loc-use',
            'cmrs_fcf-loc-use_sl-B_enc-B_rec',
            'cmrs_fcf-loc-use_sl-B_enc-B_rec_fix-B_enc_use1',
            'cmrs_fcf-loc-use_sl-B_enc-B_rec_fix-B_rec_use1',
            'cmrs_fcf-loc-use_sl-B_enc-B_rec_fix-B_enc_use1-B_rec_use1',
            'cmrs_fcf-loc_ff-cat-use',
        ]
        model_names = [
            'ID',
            'UR',
            'ED',
            'RD',
            'ED-RD',
            'CMR MP16',
        ]
    else:
        raise ValueError(f'Invalid study: {study}')
    return models, model_names


def aic(logl, n, k):
    """Akaike information criterion."""
    return -2 * logl + 2 * k + ((2 * k * (k + 1)) / (n - k - 1))


def waic(a, axis=1):
    """Akaike weights."""
    min_aic = np.expand_dims(np.min(a, axis), axis)
    delta_aic = np.exp(-0.5 * (a - min_aic))
    sum_aic = np.expand_dims(np.sum(delta_aic, axis), axis)
    return delta_aic / sum_aic


def model_comp_weights(res, stat='aic'):
    """Calculate model comparison weights."""
    # prepare statistic to calculate
    out = res.copy()
    if stat == 'aic':
        f_stat = aic
    else:
        raise ValueError(f'Invalid stat: {stat}')

    # calculate statistic and place in a pivot table
    out[stat] = f_stat(res['logl'], res['n'], res['k'])
    pivot_stat = out.reset_index().pivot(index='subject', columns='model', values=stat)

    # calculate model weights
    wstat = pivot_stat.copy()
    wstat.iloc[:, :] = waic(pivot_stat.to_numpy())
    out[f'w{stat}'] = wstat.unstack()
    return out


def generate_restricted_models():
    """Generate restricted model definitions."""
    params = ['B_enc_cat', 'B_enc_use', 'B_rec_cat', 'B_rec_use']
    fixed = [
        '-'.join([f'{p}=1' for p in c])
        for n in [1, 2, 3, 4]
        for c in combinations(params, n)
    ]

    short = ['EC', 'ED', 'RC', 'RD']
    names = ['-'.join(c) for n in [1, 2, 3, 4] for c in combinations(short, n)]
    return fixed, names


def print_restricted_models():
    """Print restricted models in a comma-separated list."""
    fixed, _ = generate_restricted_models()
    s = ','.join(fixed)
    print(s)


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


def apply_list_mask(data, mask):
    """Apply relative mask of lists to include."""
    lists = np.sort(data['list'].unique())
    include_lists = lists[mask]
    masked = data[data['list'].isin(include_lists)]
    return masked


def configure_model(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    sublayer_param,
    fixed_param,
    include,
):
    """Configure a model based on commandline input."""
    # unpack lists
    fcf_features = split_arg(fcf_features)
    ff_features = split_arg(ff_features)
    if include is not None:
        include = [int(s) for s in split_arg(include)]
    sublayer_param = split_arg(sublayer_param)
    fixed_param_list = split_arg(fixed_param)
    fixed_param = {}
    if fixed_param_list is not None:
        for expr in fixed_param_list:
            param_name, val = expr.split('=')
            fixed_param[param_name] = float(val)

    # load data to simulate
    logging.info(f'Loading data from {data_file}.')
    data = pd.read_csv(data_file)
    if include is not None:
        data = data.loc[data['subject'].isin(include)]

    # set parameter definitions based on model framework
    param_def = model_variant(
        fcf_features,
        ff_features,
        sublayers=sublayers,
        sublayer_param=sublayer_param,
        intercept=intercept,
        fixed_param=fixed_param,
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


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fcf_features")
@click.argument("ff_features")
@click.argument("res_dir", type=click.Path())
@click.option("--intercept/--no-intercept", default=False)
@click.option("--sublayers/--no-sublayers", default=False)
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
@click.option(
    "--n-sim-reps",
    "-r",
    type=int,
    default=1,
    help="number of experiment replications to simulate",
)
@click.option(
    "--include",
    "-i",
    help="dash-separated list of subject to include (default: all in data file)",
)
def fit_cmr(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    res_dir,
    intercept,
    sublayers,
    sublayer_param=None,
    fixed_param=None,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    n_sim_reps=1,
    include=None,
):
    """Run a parameter search to fit a model and simulate data."""
    os.makedirs(res_dir, exist_ok=True)
    log_file = os.path.join(res_dir, 'log_fit.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )

    # set up data and model based on script input
    data, param_def, patterns = configure_model(
        data_file,
        patterns_file,
        fcf_features,
        ff_features,
        intercept,
        sublayers,
        sublayer_param,
        fixed_param,
        include,
    )

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
    model = cmr.CMR()
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


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fcf_features")
@click.argument("ff_features")
@click.argument("res_dir", type=click.Path())
@click.option("--intercept/--no-intercept", default=False)
@click.option("--sublayers/--no-sublayers", default=False)
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
@click.option(
    "--include",
    "-i",
    help="dash-separated list of subject to include (default: all in data file)",
)
def xval_cmr(
    data_file,
    patterns_file,
    fcf_features,
    ff_features,
    res_dir,
    intercept,
    sublayers,
    sublayer_param=None,
    fixed_param=None,
    n_folds=None,
    fold_key=None,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    include=None,
):
    """Evaluate a model using cross-validation."""
    os.makedirs(res_dir, exist_ok=True)
    log_file = os.path.join(res_dir, 'log_xval.txt')
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(name)s:%(message)s',
    )

    # set up data and model based on script input
    data, param_def, patterns = configure_model(
        data_file,
        patterns_file,
        fcf_features,
        ff_features,
        intercept,
        sublayers,
        sublayer_param,
        fixed_param,
        include,
    )

    if (n_folds is None and fold_key is None) or (
        n_folds is not None and fold_key is not None
    ):
        raise ValueError('Must specify one of either n_folds or fold_key.')

    # save model information
    json_file = os.path.join(res_dir, 'xval_parameters.json')
    logging.info(f'Saving parameter definition to {json_file}.')
    param_def.to_json(json_file)

    # run individual subject fits
    n = data['subject'].nunique()
    logging.info(
        f'Running {n_reps} parameter optimization repeat(s) for {n} participant(s).'
    )
    logging.info(f'Using {n_jobs} core(s).')

    n_lists = data.groupby('subject')['list'].nunique().max()
    if fold_key is not None:
        # get folds from the events
        n_folds_all = data.groupby('subject')[fold_key].nunique()
        if len(n_folds_all.unique()) != 1:
            raise ValueError('All subjects must have same number of folds.')
        folds = data[fold_key].unique()
    else:
        # interleave folds over lists
        folds = np.arange(1, n_folds + 1)
        list_fold = np.tile(folds, int(np.ceil(n_lists / n_folds)))
    xval_list = []
    search_list = []
    model = cmr.CMR()
    for fold in folds:
        # fit the training dataset
        if fold_key is not None:
            train_data = data[data[fold_key] != fold]
        else:
            train_data = (
                data.groupby('subject')
                .apply(apply_list_mask, list_fold != fold)
                .droplevel('subject')
            )
        results = model.fit_indiv(
            train_data,
            param_def,
            patterns=patterns,
            n_jobs=n_jobs,
            method='de',
            n_rep=n_reps,
            tol=tol,
        )
        search_list.append(results)

        # evaluate on left-out fold
        best = fit.get_best_results(results)
        subj_param = best.T.to_dict()
        if fold_key is not None:
            test_data = data[data[fold_key] == fold]
        else:
            test_data = (
                data.groupby('subject')
                .apply(apply_list_mask, list_fold == fold)
                .droplevel('subject')
            )
        stats = model.likelihood(
            test_data, {}, subj_param, param_def, patterns=patterns
        )
        xval = best.copy()
        xval['logl_train'] = xval['logl']
        xval['logl_test'] = stats['logl']
        xval['n_train'] = xval['n']
        xval['n_test'] = stats['n']
        m_train = train_data.groupby('subject')['list'].nunique()
        m_test = test_data.groupby('subject')['list'].nunique()
        xval['logl_train_list'] = xval['logl_train'] / m_train
        xval['logl_test_list'] = xval['logl_test'] / m_test
        xval['m_train'] = m_train
        xval['m_test'] = m_test
        xval.drop(columns=['logl', 'n'], inplace=True)
        xval_list.append(xval)

    # cross-validation summary
    summary = pd.concat(xval_list, keys=folds)
    summary.index.rename(['fold', 'subject'], inplace=True)
    xval_file = os.path.join(res_dir, 'xval.csv')
    logging.info(f'Saving best fitting results to {xval_file}.')
    summary.to_csv(xval_file)

    # full search information
    search = pd.concat(search_list, keys=folds)
    search.index.rename(['fold', 'subject', 'rep'], inplace=True)
    search_file = os.path.join(res_dir, f'xval_search.csv')
    logging.info(f'Saving full search results to {search_file}.')
    search.to_csv(search_file)


@click.command()
@click.argument("data_file", type=click.Path(exists=True))
@click.argument("patterns_file", type=click.Path(exists=True))
@click.argument("fit_dir", type=click.Path(exists=True))
@click.option("--n-rep", "-r", type=int, default=1)
def sim_cmr(data_file, patterns_file, fit_dir, n_rep=1):
    """Run a simulation using best-fitting parameters."""
    # load trials to simulate
    data = pd.read_csv(data_file)
    study_data = data.loc[(data['trial_type'] == 'study')]

    # get model, patterns, and weights
    model = cmr.CMR()
    patterns = cmr.load_patterns(patterns_file)
    param_file = os.path.join(fit_dir, 'parameters.json')
    param_def = cmr.read_config(param_file)

    # load parameters
    fit_file = os.path.join(fit_dir, 'fit.csv')
    subj_param = read_fit_param(fit_file)

    # run simulation
    sim = model.generate(study_data, {}, subj_param, param_def, patterns, n_rep=n_rep)

    # save
    sim_file = os.path.join(fit_dir, 'sim.csv')
    sim.to_csv(sim_file, index=False)


def generate_model_name(
    fcf_features,
    ff_features,
    sublayers,
    subpar,
    fixed,
):
    """Generate standard model name from configuration."""
    if sublayers:
        res_name = 'cmrs'
    else:
        res_name = 'cmr'

    if fcf_features and fcf_features != 'none':
        res_name += f'_fcf-{fcf_features}'
    if ff_features and ff_features != 'none':
        res_name += f'_ff-{ff_features}'
    if subpar:
        res_name += f'_sl-{subpar}'
    if fixed:
        res_name += f'_fix-{fixed.replace("=", "")}'
    return res_name


def get_study_paths(study):
    """Get relevant paths based on environment."""
    study_dir = os.environ['STUDYDIR']
    if not study_dir:
        raise EnvironmentError('STUDYDIR not defined.')

    study_dir = Path(study_dir)
    if not study_dir.exists():
        raise IOError(f'Study directory does not exist: {study_dir}')

    data_file = study_dir / study / f'{study}_data.csv'
    if not data_file.exists():
        raise IOError(f'Data file does not exist: {data_file}')

    patterns_file = study_dir / study / f'{study}_patterns.hdf5'
    if not patterns_file.exists():
        raise IOError(f'Patterns file does not exist: {patterns_file}')

    return study_dir, data_file, patterns_file
