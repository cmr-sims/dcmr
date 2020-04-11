#!/usr/bin/env python
#
# Make a series of plots from a data table.

import os
import argparse
import pandas as pd
from psifr import fr
from cfr import task


def main(csv_file, out_dir):

    data = task.read_free_recall(csv_file)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if 'list_type' in data:
        mixed = data.query('list_type == "mixed"')
    else:
        mixed = data

    kwargs = {'height': 4}
    subj_kwargs = {'col': 'subject', 'col_wrap': 5, 'height': 3}

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
    parser.add_argument('csv_file')
    parser.add_argument('out_dir')
    args = parser.parse_args()
    main(args.csv_file, args.out_dir)
