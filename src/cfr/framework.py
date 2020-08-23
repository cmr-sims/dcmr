"""Fit and simulate data using CMR."""

import os
import json
import numpy as np
import pandas as pd
from cymr.parameters import Parameters
from cfr import task


class WeightParameters(Parameters):
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

    def set_weight_param(self, connect, weights, upper=100):
        """
        Add weight parameters for patterns or similarity.

        Parameters
        ----------
        connect : {'fcf', 'ff'}
            Type of connection.

        weights : list of str
            Weights to include.

        upper : float
            Upper bound of parameters beyond the first two.

        sublayers : bool, optional
            If true, weight matrices will be split into sublayers that
            will evolve context independently of one another. If false,
            weight matrices will be assigned to different regions within
            a single 'task' sublayer.
        """
        if connect == 'fcf':
            prefix = 'w'
            matrices = ['fc', 'cf']
            ssw = ' + '.join([f'wr_{name}**2' for name in weights])
            denom = f'sqrt({ssw})'
        elif connect == 'ff':
            prefix = 's'
            matrices = ['ff']
            sw = ' + '.join([f'sr_{name}' for name in weights])
            denom = f'({sw})'
        else:
            raise ValueError(f'Invalid connection type: {connect}')
        n_weight = len(weights)
        w_param = [f'{prefix}{n}' for n in range(n_weight - 1)]

        # set list of sublayers
        self.set_sublayers(f=['task'], c=['task'])

        n = 0
        m = 0
        weight_expr = []
        rescaled = {}
        for name in weights:
            param = f'{prefix}_{name}'
            raw_param = f'{prefix}r_{name}'
            if n_weight == 1:
                expr = name
            else:
                expr = f'{param} * {name}'
            weight_expr.append(expr)

            if connect == 'fcf':
                region = (('task', 'item'), ('task', name))
                for matrix in matrices:
                    w_expr = f'D{matrix} * {expr}'
                    self.set_weights(matrix, {region: w_expr})
            if n_weight == 1:
                # if only one, no weight parameter necessary
                continue

            # set up weight parameter and translate to original name
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
        self.set_dependent(rescaled)

        if connect == 'ff':
            # add compiled weight matrices
            region = ('task', 'item')
            expr = ' + '.join(weight_expr)
            self.set_free(Dff=(0, 10))
            w_expr = f'Dff * ({expr})'
            self.set_weights('ff', {region: w_expr})

    def set_weight_param_sublayer(self, weights):
        """Set sublayer parameter definitions."""
        self.set_sublayers(f=['task'], c=weights)
        for name in weights:
            region = (('task', 'item'), (name, 'item'))
            Lfc = f'Lfc_{name}'
            Lcf = f'Lcf_{name}'
            Dfc = f'Dfc_{name}'
            Dcf = f'Dcf_{name}'
            B_enc = f'B_enc_{name}'
            B_rec = f'B_rec_{name}'
            self.set_free({
                Lfc: (0, 1), Lcf: (0, 1), B_enc: (0, 1), B_rec: (0, 1)
            })
            self.set_dependent({Dfc: f'1 - {Lfc}', Dcf: f'1 - {Lcf}'})
            self.set_weights('fc', {region: f'{Dfc} * {name}'})
            self.set_weights('cf', {region: f'{Dcf} * {name}'})
            self.set_sublayer_param('c', {
                name: {
                    'B_enc': B_enc, 'B_rec': B_rec, 'Lfc': Lfc, 'Lcf': Lcf
                }
            })
        for par in ['B_enc', 'B_rec', 'Lfc', 'Lcf']:
            if par in self.free:
                del self.free[par]
        for par in ['Dfc', 'Dcf']:
            if par in self.dependent:
                del self.dependent[par]


def model_variant(fcf_features, ff_features=None, sublayers=False):
    """Define parameters for a model variant."""
    wp = WeightParameters()
    wp.set_fixed(T=0.1)
    wp.set_free(Lfc=(0, 1),
                Lcf=(0, 1),
                P1=(0, 10),
                P2=(0.1, 5),
                B_enc=(0, 1),
                B_start=(0, 1),
                B_rec=(0, 1),
                X1=(0, 1),
                X2=(0, 5))
    wp.set_dependent(Dfc='1 - Lfc',
                     Dcf='1 - Lcf')

    if fcf_features:
        if sublayers:
            wp.set_weight_param_sublayer(fcf_features)
        else:
            wp.set_weight_param('fcf', fcf_features)

    if ff_features:
        wp.set_weight_param('ff', ff_features)
        wp.set_free(Dff=(0, 10))
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

    value = {**model_def['fixed'], **model_def['free'],
             **model_def['dependent']}
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
        spec = read_model_spec(spec_file)
        spec_list.append(spec)
    model_defs = pd.concat(spec_list, keys=model_names)
    model_defs.index.rename(['model', 'param'], inplace=True)
    return model_defs


def read_model_fits(fit_dir, models, model_names=None):
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
    return res


def read_model_sims(data_file, fit_dir, models, model_names=None):
    """Read simulated data for multiple models."""
    if model_names is None:
        model_names = models

    data_list = []
    obs_data = task.read_free_recall(data_file)
    data_list.append(obs_data)
    for model in models:
        sim_file = os.path.join(fit_dir, model, 'sim.csv')
        sim_data = task.read_free_recall(sim_file)
        data_list.append(sim_data)
    data = pd.concat(data_list, axis=0, keys=['data'] + model_names)
    data.index.rename(['source', 'trial'], inplace=True)
    return data


def aic(logl, n, k):
    """Akaike information criterion."""
    return -2 * logl + 2 * k + ((2 * k * (k + 1)) / (n - k - 1))


def waic(a, axis=1):
    """Akaike weights."""
    min_aic = np.expand_dims(np.min(a, axis), axis)
    delta_aic = np.exp(-0.5 * (a - min_aic))
    sum_aic = np.expand_dims(np.sum(delta_aic, axis), axis)
    return delta_aic / sum_aic
