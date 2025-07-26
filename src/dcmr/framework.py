"""Fit and simulate data using CMR."""

import os
from pathlib import Path
import json
import logging
from itertools import combinations
from importlib import resources
import numpy as np
import pandas as pd
from cymr import cmr
from cymr import fit
from cymr.cmr import CMRParameters
from dcmr import task
from dcmr import reports


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

        Scaling parameters include a free "raw" parameter and a 
        normalized parameter that is calculated based on the set of raw
        parameters. This normalization varies depending on whether we
        are scaling similarity values or vectors.

        Parameters
        ----------
        scaling_type : {'similarity', 'vector'}
            Type of matrix to be scaled. This determines how weights
            are normalized.

        weights : list of str
            Labels of weights to include.

        upper : float
            Upper bound of parameters beyond the first two.
        
        Returns
        -------
        scaling_param : dict of (str: str)
            Expressions for parameters used to scale weights.
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
        """Set a free intercept parameter for each connection matrix."""
        intercept_param = {}
        for connect in connects:
            new_param = f'A{connect}'
            intercept_param[connect] = new_param
            self.set_free({new_param: (lower, upper)})
        return intercept_param

    def set_region_weights(self, connect, scaling_param, pre_param):
        """
        Set weights within context regions.

        Parameters
        ----------
        connect : str
            Connection matrix (fc or cf).
        
        scaling_param : dict of (str: str)
            Expressions for scaling parameters for each segment.
        
        pre_param : str
            Name of parameter that scales pre-experimental weights.
        """
        for weight, scaling in scaling_param.items():
            if scaling is not None:
                expr = f'{pre_param} * {scaling} * {weight}'
            else:
                expr = f'{pre_param} * {weight}'
            self.set_weights(connect, {(('task', 'item'), ('task', weight)): expr})

    def set_sublayer_weights(
        self, connect, scaling_param, pre_param, intercept_param=None
    ):
        """
        Set weights for context sublayers.

        Weights expressions may include the following terms:
            intercept + pre * scaling * weight
        Where terms are:
            intercept : an optional intercept parameter.
            pre : a parameter that determines the weight of 
                pre-experimental associations.
            scaling : an optional scaling parameter that determines
                the relative weighting of sublayers.
            weight : the name of the pattern to use for these weights.

        Parameters
        ----------
        connect : str
            Connection matrix (fc or cf).
        
        scaling_param : dict of (str: str)
            Expressions for scaling parameters for each sublayer. If
            None, there will be no scaling term.
        
        pre_param : str
            Name of parameter that scales pre-experimental weights.
        
        intercept_param : str or dict of (str: str)
            Expression to apply to all sublayers or dictionary with
            an expression for each sublayer.
        """
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
        """
        Set item-item weights.

        Weights expressions may include an intercept term and additive
        terms for each set of weights to be included, optionally scaled
        by scaling parameters:
            intercept + scaling1 * weight1 + scaling2 * weight2 + ...

        Parameters
        ----------
        scaling_param : dict of (str: str)
            Expressions for scaling parameters for each set of weights.
        
        pre_param : str
            Name of parameter that scales pre-experimental weights.
        
        intercept_param : str or dict of (str: str)
            Expression to apply to all sublayers or dictionary with
            an expression for each sublayer.
        """
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
        """
        Set dependent sublayer parameters for learning.

        Learning parameters will be set to vary by sublayer. The
        pre-experimental parameters will be modified to be dependent
        on the new sublayer-specific learning parameters, so they will
        be set to (1 - L) for that sublayer.

        Parameters
        ----------
        L_name : str
            Name of the original learning parameter.
        
        D_name : str
            Name of the original pre-experimental parameter.
        
        Returns
        -------
        L_param : dict of str
            The new sublayer-specific learning parameters.
        
        D_param : dict of str
            The new sublayer-specific pre-experimental parameters.
        """
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
        """
        Set scaling of sublayer learning rates.

        Apply scaling to learning rate parameters. Sets a dependent
        parameter that is scaled and sets that as a sublayer parameter.
        
        Parameters
        ----------
        scaling_param : dict of (str: str)
            Scaling parameter name for each sublayer.
        
        suffix : dict of (str: str)
            Suffix for sublayer-specific learning parameters Lfc and 
            Lcf.
        """
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
        """
        Set sublayer parameters to be free.

        Parameters must already be defined as free. Specified 
        parameters will be modified to vary by sublayer, with the same
        range as the original parameter. The original parameter will be
        removed from the list of free parameters.

        Parameters
        ----------
        param_names : list of str
            Parameters to free to vary by sublayer.
        
        suffix : str
            Suffix to add for modified parameters.
        """
        for param in param_names:
            # make a copy of the base parameter for each sublayer
            for weight in self.sublayers['c']:
                param_name = f'{param}_{weight}'
                if suffix is not None:
                    param_name += suffix
                if param in self.free:
                    self.set_free({param_name: self.free[param]})
                self.set_sublayer_param('c', weight, {param: param_name})

            # remove the base parameter from the list of free variables
            if param in self.free:
                del self.free[param]
    
    def expand_sublayer_param(self, param_names=None):
        """
        Expand sublayer parameters to all have the same parameters defined.

        Update sublayer parameters to ensure that, if a given parameter
        is set for any one sublayer, that parameter will be defined 
        explicitly for all sublayers. If a sublayer parameter has not
        been defined previously, the global value for that parameter
        will be used.

        Parameters
        ----------
        param_names : list of str, optional
            If defined, only the specified parameter names will be
            set for each sublayer. If None, all parameters defined
            for at least one sublayer will be set.
        """
        if param_names is None:
            # get all parameters that are defined for any sublayer
            param_names = set()
            for weight, params in self.sublayer_param['c'].items():
                param_names.update(params.keys())

        # if a parameter is not defined for a sublayer, set it to the
        # global value of that parameter
        for par in param_names:
            for weight in self.sublayers['c']:
                if par not in self.sublayer_param['c'][weight]:
                    self.set_sublayer_param('c', weight, {par: par})


def model_variant(
    fcf_features,
    ff_features=None,
    sublayers=False,
    scaling=True,
    sublayer_param=None,
    intercept=False,
    distraction=False,
    special_sublayers=None,
    fixed_param=None,
    free_param=None,
    dependent_param=None,
    dynamic_param=None,
):
    """
    Define parameters for a model variant.

    Parameters are determined in the following order:

        dependent evaluated based on other static parameters
        dynamic evaluated based on static parameters and data
        dynamic set for the current list
        dependent evaluated given current dynamic parameters
        sublayer parameters evaluated given current parameters

    Parameters
    ----------
    fcf_features : list of str
        Features to include in the FC and CF matrices connecting the
        item and context layers.
    
    ff_features : list of str
        Features to include in the FF matrices connecting items to
        other items.
    
    sublayers : bool
        If True, features will be represented in different sublayers,
        which may vary in their dynamics and are normalized separately.
        If False, features will be represented in different segments of
        a single layer of context.
    
    sublayer_param : list of str
        Names of parameters that will vary by sublayer. If Lfc and/or
        Lcf are included, the dependent Dfc and Dcf parameters will 
        also be adjusted; for example, Lfc might be split into Lfc_loc
        and Lfc_cat, Dfc_loc will be defined as 1 - Lfc_loc, and 
        Dcf_loc will be defined as 1 - Lcf_loc.
    
    intercept : bool
        If True, an intercept term will be included in the FF matrix to
        add a baseline level of cuing strength for all included items.

    distraction : bool
        If True, distraction units will be added to the network, which
        may be used to disrupt context before and after each item
        presentation.

    special_sublayers : list of str
        Special sublayers to include in the network. May include 'list'
        for a static list context.

    fixed_param : dict of (str: float)
        Parameters to set to a specified fixed value. Any free 
        parameters of the same name will be removed from the free 
        parameter list.
    
    free_param : dict of (str: (float, float))
        Parameters to set as free parameters, with the corresponding
        range of allowed values. Any fixed parameters of the same name
        will be removed from the fixed parameter list.
    
    dependent_param : dict of (str: str)
        Parameters to set as dependent on other parameters, with an
        expression defining how they should be evaluated. Expressions
        may call NumPy functions.
    
    dynamic_param : dict of ((str, str): str)
        Parameters to set as dynamic values that depend on variables in
        the data. Parameters are listed by trial type ("study" or 
        "recall") and scope ("trial" or "list"), with each dictionary
        key being a (trial_type, scope) tuple. Values indicate 
        expressions to define the dynamic parameter, which may 
        reference other parameters, data columns, and NumPy functions.

    Returns
    -------
    wp : WeightParameters
        A parameter definition object defining the specified model 
        variant. Can be used to run parameter searches to fit the model
        variant. Can also be used with specific fitted or manually set
        parameters to evaluate likelihood or run a simulation.
    """
    wp = WeightParameters()
    wp.set_options(distraction=distraction)
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

    # add free parameters and/or modify free parameters
    if free_param is not None:
        wp.set_free(free_param)
        for param_name in free_param.keys():
            if param_name in fixed_param:
                del wp.fixed[param_name]
    
    # add dependent parameters
    if dependent_param is not None:
        wp.set_dependent(dependent_param)
        for param_name in dependent_param.keys():
            if param_name in wp.free:
                del wp.free[param_name]
    
    # add dynamic parameters
    if dynamic_param is not None:
        for (trial_type, scope), dyn in dynamic_param.items():
            wp.set_dynamic(trial_type, scope, dyn)

    if intercept:
        if 'Aff' in wp.free:
            Aff = wp.free['Aff']
        else:
            Aff = (-1, 1)
        intercept_param = wp.set_intercept_param(['ff'], *Aff)
    else:
        intercept_param = None

    if fcf_features:
        if special_sublayers is None:
            special_sublayers = []

        # set global weight scaling
        if scaling:
            scaling_param = wp.set_scaling_param('vector', fcf_features)
        else:
            scaling_param = {key: None for key in fcf_features}

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

            if 'list' in special_sublayers:
                # add a static list context sublayer
                wp.sublayers['c'].append('list')

                # set list sublayer parameters
                list_context_param = {
                    'Lfc': 0.5, 
                    'Lcf': 'Acf', 
                    'B_enc': 0, 
                    'B_rec': 0, 
                    'B_start': 0, 
                }
                if distraction:
                    list_context_param.update({'B_distract': 0, 'B_retention': 0})
                if 'Acf' not in wp.free:
                    wp.set_free(Acf=(0, 1))
                wp.set_sublayer_param('c', 'list', list_context_param)

                # set corresponding other sublayer parameters if necessary
                wp.expand_sublayer_param(list_context_param.keys())

                # set pre-experimental weights
                expr1 = 'ones((loc.shape[0], 1))'
                expr0 = 'zeros((loc.shape[0], 1))'
                wp.set_weights('fc', {(('task', 'item'), ('list', 'item')): expr1})
                wp.set_weights('cf', {(('task', 'item'), ('list', 'item')): expr0})
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
            if param_name in wp.free:
                del wp.free[param_name]
    return wp


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


def generate_model_name(
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    subpar,
    fixed,
    free,
    dependent,
):
    """Generate standard model name from configuration."""
    if sublayers:
        res_name = 'cmrs'
    else:
        res_name = 'cmr'
    
    if not scaling:
        res_name += 'n'

    if intercept:
        res_name += 'i'

    if fcf_features and fcf_features != 'none':
        res_name += f'_fcf-{fcf_features}'
    if ff_features and ff_features != 'none':
        res_name += f'_ff-{ff_features}'
    if subpar:
        res_name += f'_sl-{subpar}'
    if fixed:
        res_name += f'_fix-{fixed.replace("=", "")}'
    if free:
        res_name += f'_free-{free.replace("=", "").replace(":", "to")}'
    if dependent:
        res_name += f'_dep-{dependent.replace("=", "")}'
    return res_name


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
    data_file, 
    fit_dir, 
    models, 
    model_names=None, 
    block=False, 
    block_category=False, 
    data_first=False,
):
    """Read simulated data for multiple models."""
    if model_names is None:
        model_names = models

    obs_data = task.read_free_recall(
        data_file, block=block, block_category=block_category
    )
    if data_first:
        data_list = [obs_data]
        keys = ['Data'] + model_names
    else:
        data_list = []
        keys = model_names + ['Data']

    for model in models:
        sim_file = os.path.join(fit_dir, model, 'sim.csv')
        sim_data = task.read_free_recall(
            sim_file, block=block, block_category=block_category
        )
        data_list.append(sim_data)

    if not data_first:
        data_list.append(obs_data)

    data = pd.concat(data_list, axis=0, keys=keys)
    data.index.rename(['source', 'trial'], inplace=True)
    return data


def get_sim_models(study, model_set, included=None):
    """Get a list of models for a study."""
    list_file = resources.files('dcmr') / 'models' / f'{study}.json'
    with open(list_file, 'r') as f:
        model_list = json.load(f)
        if included is not None:
            model_dict = {
                s[model_set]: s['full']
                for short_name, s in model_list.items()
                if model_set in s and s[model_set] in included
            }
        else:
            model_dict = {
                s[model_set]: s['full']
                for short_name, s in model_list.items()
                if model_set in s
            }
        model_names = list(model_dict.keys())
        models = list(model_dict.values())
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


def compare_fit(means):
    """Compare model fit to mean measures."""
    if "Data" not in means.columns:
        raise ValueError("Must have a Data column.")
    models = [n for n in means.columns if n != "Data"]
    output = pd.DataFrame(columns=models, index=["rmsd"])

    # rmsd
    for model in models:
        output.loc["rmsd", model] = np.sqrt(np.mean((means["Data"] - means[model]) ** 2))
    return output


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


def apply_list_mask(data, mask):
    """Apply relative mask of lists to include."""
    lists = np.sort(data['list'].unique())
    include_lists = lists[mask]
    masked = data[data['list'].isin(include_lists)]
    return masked


def run_fit(
    res_dir, 
    data, 
    param_def, 
    patterns, 
    n_jobs, 
    n_reps, 
    tol=0.00001, 
    init='latinhypercube',
    n_sim_reps=1, 
    study_keys=None,
    recall_keys=None,
    category=None,
    similarity=None,
):
    """
    Fit parameters to individual subjects of a dataset.
    
    Given data, parameter configuration, and patterns, estimate 
    parameters to fit the model to the dataset. Then generate simulated
    data for analysis. Finally, create a fit report with common 
    statistics, parameters, and model snapshots.

    Parameters
    ----------
    res_dir : str
        Path to directory to save results.
    
    data : pandas.DataFrame
        Dataset in Psifr format to be fitted.
    
    param_def : framework.WeightParameters
        Parameter definition object with model configuration.
    
    patterns : dict
        Pattern definitions in CyMR format.
    
    n_jobs : int
        Number of cores to use during search.
    
    n_reps : int
        Number of times to replicate the search. Best-fitting 
        parameters will be selected from each subject's search.

    tol : float
        Tolerance for terminating the search, based on likelihood.

    init : str
        Method for initializing the search.

    n_sim_reps : int
        Number of times to replicate simulation of the dataset using
        the best-fitting values.
    
    study_keys : list of str
        Columns of data to include in the study data during fitting and
        simulations.

    recall_keys : list of str
        Columns of data to include in the recall data during 
        simulations.
    
    category : bool
        If True, the fit report will include category-related analyses.
        If None, will attempt to determine from the data.
    
    similarity : bool
        If True, the fit report will include semantic similarity 
        analyses. If None, will attempt to determine from the data.
    """
    if study_keys is not None:
        study_keys = list(study_keys)
    if recall_keys is not None:
        recall_keys = list(recall_keys)

    # save model information
    json_file = os.path.join(res_dir, 'parameters.json')
    logging.info(f'Saving parameter definition to {json_file}.')
    param_def.to_json(json_file)

    # run individual subject fits
    n = data['subject'].nunique()
    method = 'de'
    logging.info(
        f'Running {n_reps} parameter optimization repeat(s) for {n} participant(s).'
    )
    logging.info(
        f'Using search method {method} with parameter initialization method {init}.'
    )
    logging.info(f'Using {n_jobs} core(s).')
    model = cmr.CMR()
    results = model.fit_indiv(
        data,
        param_def,
        patterns=patterns,
        n_jobs=n_jobs,
        method=method,
        n_rep=n_reps,
        tol=tol,
        study_keys=study_keys,
        recall_keys=recall_keys,
        init=init,
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
        study_keys=study_keys,
        recall_keys=recall_keys,
    )
    sim_file = os.path.join(res_dir, 'sim.csv')
    logging.info(f'Saving simulated data to {sim_file}.')
    sim.to_csv(sim_file, index=False)

    # make a report of the fit
    reports.plot_fit(
        data, 
        sim, 
        {},
        subj_param,
        param_def,
        patterns, 
        best.reset_index(),
        res_dir, 
        study_keys=study_keys, 
        category=category,
        similarity=similarity,
    )
    return best


def run_xval(
    res_dir,
    data,
    param_def,
    patterns,
    n_folds=None,
    fold_key=None,
    n_reps=1,
    n_jobs=1,
    tol=0.00001,
    init='latinhypercube',
    study_keys=None,
    recall_keys=None,
):
    """
    Evaluate a model using cross-validation.
    
    Given data, parameter configuration, and patterns, run a cross-
    validation analysis. This involves splitting the data into groups
    of lists, estimating parameters based on a subset of lists, and
    evaluating the likelihood for those parameters on a left-out set of
    lists. Results include best-fitting parameters for each cross-
    validation fold, training-set likelihood, and testing-set 
    likelihood.

    Parameters
    ----------
    res_dir : str
        Path to directory to save results.
    
    data : pandas.DataFrame
        Dataset in Psifr format to be fitted.
    
    param_def : framework.WeightParameters
        Parameter definition object with model configuration.
    
    patterns : dict
        Pattern definitions in CyMR format.
    
    n_folds : int
        Number of folds to split lists into. Must specify either this
        or fold_key. Folds will be interleaved across lists.
    
    fold_key : str
        Column of data to use when dividing lists. Each unique value
        will result in one fold.
    
    n_reps : int
        Number of times to replicate the search. Best-fitting 
        parameters will be selected from each subject's search.

    n_jobs : int
        Number of cores to use during search.
    
    tol : float
        Tolerance for terminating the search, based on likelihood.

    init : str
        Method for initializing the search.

    study_keys : list of str
        Columns of data to include in the study data during fitting and
        simulations.

    recall_keys : list of str
        Columns of data to include in the recall data during 
        simulations.
    """
    if study_keys is not None:
        study_keys = list(study_keys)
    if recall_keys is not None:
        recall_keys = list(recall_keys)

    # check cross-validation settings
    if (n_folds is None and fold_key is None) or (
        n_folds is not None and fold_key is not None
    ):
        raise ValueError('Must specify one of either n_folds or fold_key.')

    # save model information
    json_file = os.path.join(res_dir, 'xval_parameters.json')
    logging.info(f'Saving parameter definition to {json_file}.')
    param_def.to_json(json_file)

    n = data['subject'].nunique()
    if fold_key is not None:
        # get folds from the events
        n_folds_all = data.groupby('subject')[fold_key].nunique()
        if len(n_folds_all.unique()) != 1:
            raise ValueError('All subjects must have same number of folds.')
        folds = data[fold_key].unique()
        logging.info(f'Running {len(folds)} cross-validation folds over {fold_key} for {n} participants.')
    else:
        # interleave folds over lists
        folds = np.arange(1, n_folds + 1)
        n_lists = data.groupby('subject')['list'].nunique().max()
        list_fold = np.tile(folds, int(np.ceil(n_lists / n_folds)))
        logging.info(f'Running {len(folds)} cross-validation folds for {n} participants.')
    
    # run cross-validation
    method = 'de'
    logging.info(f'Running {n_reps} parameter optimization repeat(s).')
    logging.info(
        f'Using search method {method} with parameter initialization method {init}.'
    )
    logging.info(f'Using {n_jobs} core(s).')
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
            method=method,
            n_rep=n_reps,
            tol=tol,
            init=init,
            study_keys=study_keys,
            recall_keys=recall_keys,
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
            test_data, 
            {}, 
            subj_param, 
            param_def, 
            patterns=patterns, 
            study_keys=study_keys, 
            recall_keys=recall_keys,
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
