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

    def set_region_weights(self, connect, scaling_param, pre_param):
        """Sets weights within sublayer regions."""
        for weight, scaling in scaling_param.items():
            if scaling is not None:
                expr = f'{pre_param} * {scaling} * {weight}'
            else:
                expr = f'{pre_param} * {weight}'
            self.set_weights(connect, {
                (('task', 'item'), ('task', weight)): expr
            })

    def set_sublayer_weights(self, connect, scaling_param, pre_param):
        """Set weights for different sublayers."""
        for weight, scaling in scaling_param.items():
            if isinstance(pre_param, str):
                pre = pre_param
            else:
                pre = pre_param[weight]
            if scaling is not None:
                expr = f'{pre} * {scaling} * {weight}'
            else:
                expr = f'{pre} * {weight}'
            self.set_weights(connect, {
                (('task', 'item'), (weight, 'item')): expr
            })

    def set_item_weights(self, scaling_param, pre_param):
        """Set item-item weights."""
        weight_expr = []
        for weight, scaling in scaling_param.items():
            if scaling is not None:
                expr = f'{scaling} * {weight}'
            else:
                expr = weight
            weight_expr.append(expr)
        expr = ' + '.join(weight_expr)
        w_expr = f'{pre_param} * ({expr})'
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


def model_variant(fcf_features, ff_features=None, sublayers=False,
                  sublayer_param=None):
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
        wp.set_item_weights(scaling_param, 'Dff')
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
