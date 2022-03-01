#!/usr/bin/env python
#
# Plot data and model fit.

import os
import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
from psifr import fr
from cymr import network
from cfr import task
from cfr import figures


def main(data_file, patterns_file, fit_dir, ext='svg'):
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
    data = task.read_free_recall(data_file, block=False, block_category=False)
    logging.info(f'Loading simulation from {sim_file}.')
    sim = task.read_free_recall(sim_file, block=False, block_category=False)

    # prep semantic similarity
    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = network.load_patterns(patterns_file)
    rsm = patterns['similarity']['use']
    edges = np.linspace(0.05, 0.95, 10)

    # concatenate for analysis
    full = pd.concat((data, sim), axis=0, keys=['Data', 'Model'])
    full.index.rename(['source', 'trial'], inplace=True)

    # make plots
    fig_dir = os.path.join(fit_dir, 'figs')
    kwargs = {'ext': ext}
    logging.info(f'Saving figures to {fig_dir}.')

    # scalar stats
    logging.info('Plotting fits to individual scalar statistics.')
    figures.plot_fit_scatter(
        full,
        'source',
        'use_rank',
        fr.distance_rank,
        {'index_key': 'item_index', 'distances': rsm},
        'rank',
        fig_dir,
        **kwargs,
    )
    figures.plot_fit_scatter(
        full,
        'source',
        'use_rank_within',
        fr.distance_rank,
        {
            'index_key': 'item_index',
            'distances': rsm,
            'test_key': 'category',
            'test': lambda x, y: x == y,
        },
        'rank',
        fig_dir,
        **kwargs,
    )
    figures.plot_fit_scatter(
        full,
        'source',
        'use_rank_across',
        fr.distance_rank,
        {
            'index_key': 'item_index',
            'distances': rsm,
            'test_key': 'category',
            'test': lambda x, y: x != y,
        },
        'rank',
        fig_dir,
        **kwargs,
    )
    figures.plot_fit_scatter(
        full, 'source', 'lag_rank', fr.lag_rank, {}, 'rank', fig_dir, **kwargs
    )
    figures.plot_fit_scatter(
        full,
        'source',
        'lag_rank_within',
        fr.lag_rank,
        {'test_key': 'category', 'test': lambda x, y: x == y},
        'rank',
        fig_dir,
        **kwargs,
    )
    figures.plot_fit_scatter(
        full,
        'source',
        'lag_rank_across',
        fr.lag_rank,
        {'test_key': 'category', 'test': lambda x, y: x != y},
        'rank',
        fig_dir,
        **kwargs,
    )
    figures.plot_fit_scatter(
        full,
        'source',
        'cat_crp',
        fr.category_crp,
        {'category_key': 'category'},
        'prob',
        fig_dir,
        **kwargs,
    )

    # curves
    logging.info('Plotting fits to curves.')
    figures.plot_fit(
        full,
        'source',
        'use_crp',
        fr.distance_crp,
        {'index_key': 'item_index', 'distances': rsm, 'edges': edges},
        'prob',
        fr.plot_distance_crp,
        {'min_samples': 10},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full,
        'source',
        'use_crp_within',
        fr.distance_crp,
        {
            'index_key': 'item_index',
            'distances': rsm,
            'edges': edges,
            'test_key': 'category',
            'test': lambda x, y: x == y,
        },
        'prob',
        fr.plot_distance_crp,
        {'min_samples': 10},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full,
        'source',
        'use_crp_across',
        fr.distance_crp,
        {
            'index_key': 'item_index',
            'distances': rsm,
            'edges': edges,
            'test_key': 'category',
            'test': lambda x, y: x != y,
        },
        'prob',
        fr.plot_distance_crp,
        {'min_samples': 10},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full, 'source', 'spc', fr.spc, {}, 'recall', fr.plot_spc, {}, fig_dir, **kwargs
    )
    figures.plot_fit(
        full,
        'source',
        'pfr',
        lambda x: fr.pnr(x).query('output == 1'),
        {},
        'prob',
        fr.plot_spc,
        {},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full,
        'source',
        'lag_crp',
        fr.lag_crp,
        {},
        'prob',
        fr.plot_lag_crp,
        {},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full,
        'source',
        'lag_crp_within',
        fr.lag_crp,
        {'test_key': 'category', 'test': lambda x, y: x == y},
        'prob',
        fr.plot_lag_crp,
        {},
        fig_dir,
        **kwargs,
    )
    figures.plot_fit(
        full,
        'source',
        'lag_crp_across',
        fr.lag_crp,
        {'test_key': 'category', 'test': lambda x, y: x != y},
        'prob',
        fr.plot_lag_crp,
        {},
        fig_dir,
        **kwargs,
    )

    # report
    curves = [
        'spc',
        'pfr',
        'lag_crp',
        'lag_crp_within',
        'lag_crp_across',
        'use_crp',
        'use_crp_within',
        'use_crp_across',
    ]
    points = {
        'lag_rank': ['lag_rank', 'lag_rank_within', 'lag_rank_across'],
        'cat_crp': ['cat_crp'],
        'use_rank': ['use_rank', 'use_rank_within', 'use_rank_across'],
    }
    os.chdir(fit_dir)
    figures.render_fit_html('.', curves, points)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('patterns_file')
    parser.add_argument('fit_dir')
    parser.add_argument(
        '--ext', '-e', default="svg", help="figure file type (default: svg)"
    )
    args = parser.parse_args()

    main(args.data_file, args.patterns_file, args.fit_dir, ext=args.ext)
