#!/usr/bin/env python
#
# Make a series of plots from a data table.

import os
import argparse
import numpy as np
import pandas as pd
from psifr import fr
from cfr import task


def semantic_crp_plots(data, sim_file, out_dir, kwargs, subj_kwargs,
                       bin_file=None):
    """Make semantic crp plots."""

    # read pool information
    sim = task.read_similarity(sim_file)
    data = task.set_item_index(data, sim['item'])
    if bin_file is None:
        edges = np.arange(0, 1.01, .05)
        centers = None
    else:
        raise ValueError('Reading bin files currently unsupported.')

    # semantic crps
    crp = fr.distance_crp(data, 'item_index', sim['similarity'],
                          edges, centers)
    g = fr.plot_distance_crp(crp, min_samples=10, **kwargs)
    g.savefig(os.path.join(out_dir, 'sem_crp.pdf'))

    g = fr.plot_distance_crp(crp, min_samples=10, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'sem_crp_subject.pdf'))


def main(csv_file, out_dir, sim_file=None, bin_file=None):

    data = task.read_free_recall(csv_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if 'list_type' in data:
        mixed = data.query('list_type == "mixed"').copy()
    else:
        mixed = data

    kwargs = {'height': 4}
    subj_kwargs = {'col': 'subject', 'col_wrap': 5, 'height': 3}

    if sim_file is not None:
        semantic_crp_plots(mixed, sim_file, out_dir,
                           kwargs, subj_kwargs, bin_file=bin_file)

    # spc
    recall = fr.spc(mixed)
    g = fr.plot_spc(recall)
    g.savefig(os.path.join(out_dir, 'spc.pdf'))

    g = fr.plot_spc(recall, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'spc_subject.pdf'))

    # spc by category
    recall = mixed.groupby('category').apply(fr.spc)
    g = fr.plot_spc(recall, hue='category', **kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'spc_category.pdf'))

    g = fr.plot_spc(recall, hue='category', **subj_kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'spc_category_subject.pdf'))

    # crp
    crp = fr.lag_crp(mixed)
    g = fr.plot_lag_crp(crp, **kwargs)
    g.savefig(os.path.join(out_dir, 'crp.pdf'))

    g = fr.plot_lag_crp(crp, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'crp_subject.pdf'))

    # within-category crp by category
    categories = mixed['category'].cat.categories
    cat_crp = [fr.lag_crp(mixed, item_query=f'category == "{category}"')
               for category in categories]
    crp = pd.concat(cat_crp, keys=categories, axis=0)
    crp.index = crp.index.set_names('category', level=0)
    g = fr.plot_lag_crp(crp, hue='category', **kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_within_category.pdf'))

    g = fr.plot_lag_crp(crp, hue='category', **subj_kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_within_category_subject.pdf'))

    # within-category crp
    crp = fr.lag_crp(mixed, test_key='category', test=lambda x, y: x == y)
    g = fr.plot_lag_crp(crp, **kwargs)
    g.savefig(os.path.join(out_dir, 'crp_within.pdf'))

    g = fr.plot_lag_crp(crp, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'crp_within_subject.pdf'))

    # across-category crp
    crp = fr.lag_crp(mixed, test_key='category', test=lambda x, y: x != y)
    g = fr.plot_lag_crp(crp, **kwargs)
    g.savefig(os.path.join(out_dir, 'crp_across.pdf'))

    g = fr.plot_lag_crp(crp, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'crp_across_subject.pdf'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('csv_file', help="csv file with free recall data")
    parser.add_argument('out_dir', help="directory to save figures")
    parser.add_argument('--similarity', '-s',
                        help="MAT-file with similarity matrix")
    args = parser.parse_args()
    main(args.csv_file, args.out_dir, args.similarity)
