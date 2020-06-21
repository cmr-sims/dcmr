#!/usr/bin/env python
#
# Plot data and model fit.

import os
import argparse
import numpy as np
import pandas as pd
from psifr import fr
from cymr import network
from cfr import task
from cfr import figures


def main(data_file, patterns_file, fit_dir):
    # load data and simulated data
    sim_file = os.path.join(fit_dir, 'sim.csv')
    data = task.read_free_recall(data_file)
    sim = task.read_free_recall(sim_file)

    # prep semantic similarity
    patterns = network.load_patterns(patterns_file)
    rsm = patterns['similarity']['use']
    edges = np.linspace(.05, .95, 10)

    # concatenate for analysis
    full = pd.concat((data, sim), axis=0, keys=['Data', 'Model'])
    full.index.rename(['source', 'trial'], inplace=True)

    # make plots
    fig_dir = os.path.join(fit_dir, 'figs')

    # scalar stats
    figures.plot_fit_scatter(
        full, 'source', 'use_rank', fr.distance_rank,
        {'index_key': 'item_index', 'distances': rsm}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'use_rank_within', fr.distance_rank,
        {'index_key': 'item_index', 'distances': rsm,
         'test_key': 'category', 'test': lambda x, y: x == y}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'use_rank_across', fr.distance_rank,
        {'index_key': 'item_index', 'distances': rsm,
         'test_key': 'category', 'test': lambda x, y: x != y}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'lag_rank', fr.lag_rank, {}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'lag_rank_within', fr.lag_rank,
        {'test_key': 'category', 'test': lambda x, y: x == y}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'lag_rank_across', fr.lag_rank,
        {'test_key': 'category', 'test': lambda x, y: x != y}, 'rank', fig_dir
    )
    figures.plot_fit_scatter(
        full, 'source', 'cat_crp', fr.category_crp,
        {'category_key': 'category'}, 'prob', fig_dir
    )

    # curves
    figures.plot_fit(
        full, 'source', 'use_crp',
        lambda x: fr.distance_crp(x, 'item_index', rsm, edges), {}, 'prob',
        fr.plot_distance_crp, {'min_samples': 10}, fig_dir
    )
    figures.plot_fit(
        full, 'source', 'spc', fr.spc, {}, 'recall',
        fr.plot_spc, {}, fig_dir
    )
    figures.plot_fit(
        full, 'source', 'pfr',
        lambda x: fr.pnr(x).query('output == 1'), {}, 'prob',
        fr.plot_spc, {}, fig_dir
    )
    figures.plot_fit(
        full, 'source', 'lag_crp', fr.lag_crp, {}, 'prob',
        fr.plot_lag_crp, {}, fig_dir
    )
    figures.plot_fit(
        full, 'source', 'lag_crp_within', fr.lag_crp,
        {'test_key': 'category', 'test': lambda x, y: x == y},
        'prob', fr.plot_lag_crp, {}, fig_dir
    )
    figures.plot_fit(
        full, 'source', 'lag_crp_across', fr.lag_crp,
        {'test_key': 'category', 'test': lambda x, y: x != y},
        'prob', fr.plot_lag_crp, {}, fig_dir
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('patterns_file')
    parser.add_argument('fit_dir')
    args = parser.parse_args()

    main(args.data_file, args.patterns_file, args.fit_dir)
