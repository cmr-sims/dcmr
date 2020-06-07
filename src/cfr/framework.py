"""Fit and simulate data using CMR."""

import json
import pandas as pd
from cymr.fit import Parameters


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

    def add_weight_param(self, connect, weights, upper=100):
        """
        Add weight parameters for patterns or similarity.

        Parameters
        ----------
        connect : {'fdf', 'ff'}
            Type of connection.

        weights : list of str
            Weights to include.

        upper : float
            Upper bound of parameters beyond the first two.
        """
        if connect == 'fcf':
            prefix = 'w'
        elif connect == 'ff':
            prefix = 's'
        else:
            raise ValueError(f'Invalid connection type: {connect}')
        n_weight = len(weights)
        w_param = [f'{prefix}{n}' for n in range(n_weight - 1)]

        n = 0
        m = 0
        for name in weights:
            param = f'{prefix}_{name}'
            self.add_weights(connect, {name: param})
            if n_weight == 1:
                # if only one, no weight parameter necessary
                self.add_fixed({param: 1})
                continue

            # set up weight parameter and translate to original name
            if n == 0:
                ref_param = w_param[m]
                self.add_free({ref_param: (0, 1)})
                self.add_dependent({param: ref_param})
                m += 1
            elif n == 1:
                self.add_dependent({param: f'1 - {ref_param}'})
            else:
                new_param = w_param[m]
                self.add_free({new_param: (0, upper)})
                self.add_dependent({param: new_param})
                m += 1
            n += 1


def model_variant(fcf_features, ff_features=None):
    """Define parameters for a model variant."""
    wp = WeightParameters()
    wp.add_fixed(Afc=0,
                 Acf=0,
                 Aff=0,
                 Dff=1,
                 T=0.1)
    wp.add_free(Lfc=(0, 1),
                Lcf=(0, 1),
                P1=(0, 10),
                P2=(0.1, 5),
                B_enc=(0, 1),
                B_start=(0, 1),
                B_rec=(0, 1),
                X1=(0, 1),
                X2=(0, 5))
    wp.add_dependent(Dfc='1 - Lfc',
                     Dcf='1 - Lcf')

    if fcf_features:
        wp.add_weight_param('fcf', fcf_features)

    if ff_features:
        wp.add_weight_param('ff', ff_features)
        del wp.fixed['Dff']
        wp.add_free(Dff=(0, 10))
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
