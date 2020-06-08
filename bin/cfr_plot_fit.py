#!/usr/bin/env python
#
# Plot data and model fit.

import os
import argparse
import pandas as pd
from psifr import fr
from cfr import task
from cfr import figures


def main(data_file, fit_dir):
    # load data and simulated data
    sim_file = os.path.join(fit_dir, 'sim.csv')
    data = task.read_free_recall(data_file)
    sim = task.read_free_recall(sim_file)

    # concatenate for analysis
    full = pd.concat((data, sim), axis=0, keys=['Data', 'Model'])
    full.index.rename(['source', 'trial'], inplace=True)

    # make plots
    fig_dir = os.path.join(fit_dir, 'figs')
    figures.plot_fit(full, 'source', 'spc', fr.spc, {}, fr.plot_spc, fig_dir)
    figures.plot_fit(full, 'source', 'crp', fr.lag_crp, {},
                     fr.plot_lag_crp, fig_dir)
    figures.plot_fit(full, 'source', 'crp_within', fr.lag_crp,
                     {'test_key': 'category', 'test': lambda x, y: x == y},
                     fr.plot_lag_crp, fig_dir)
    figures.plot_fit(full, 'source', 'crp_across', fr.lag_crp,
                     {'test_key': 'category', 'test': lambda x, y: x != y},
                     fr.plot_lag_crp, fig_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('fit_dir')
    args = parser.parse_args()

    main(args.data_file, args.fit_dir)
