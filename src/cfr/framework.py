"""Fit and simulate data using CMR."""

import numpy as np
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

    def add_weights(self, weights, prefix='w', upper=100):
        """
        Add weight parameters for patterns or similarity.

        Parameters
        ----------
        weights : dict of (str: bool)
            Weights to include in the parameterization, and whether to
            allow it to be free and nonzero.

        prefix : str
            Prefix for weight parameters.

        upper : float
            Upper bound of parameters beyond the first two.
        """
        n_weight = np.count_nonzero([include for include in weights.values()])
        w_param = [f'{prefix}{n}' for n in range(n_weight - 1)]

        n = 0
        m = 0
        for name, include in weights.items():
            param = f'{prefix}_{name}'
            if not include:
                # this pattern is turned off
                self.add_fixed({param: 0})
            else:
                if n_weight == 1:
                    # if only one, no weight parameter necessary
                    self.add_fixed({param: 1})
                    continue

                # set up weight parameter and translate to original name
                if n == 0:
                    ref_param = w_param[m]
                    self.add_free({ref_param: (0, 1)})
                    self.add_dependent({param: lambda par: par[ref_param]})
                    m += 1
                elif n == 1:
                    self.add_dependent({param: lambda par: 1 - par[ref_param]})
                else:
                    new_param = w_param[m]
                    self.add_free({new_param: (0, upper)})
                    self.add_dependent({param: lambda par: par[new_param]})
                    m += 1
                n += 1
