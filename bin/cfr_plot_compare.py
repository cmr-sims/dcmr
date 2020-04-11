#!/usr/bin/env python
#
# Plot comparisons of datasets or conditions.

import os
import argparse
import seaborn as sns
from psifr import fr
from cfr import task


def main(out_dir, csv_file, comp_csv=None, group_var=None):

    data = task.read_free_recall(csv_file)

    n_group = len(data[group_var].unique())
    palette = sns.color_palette('viridis', n_group)

    kwargs = {'height': 4}
    subj_kwargs = {'col': 'subject', 'col_wrap': 5, 'height': 3}

    # spc by condition
    recall = data.groupby(group_var).apply(fr.spc)
    g = fr.plot_spc(recall, hue=group_var, palette=palette, **kwargs)
    g.savefig(os.path.join(out_dir, 'spc.pdf'))

    g = fr.plot_spc(recall, hue=group_var, palette=palette, **subj_kwargs)
    g.savefig(os.path.join(out_dir, 'spc_subject.pdf'))

    # crp by condition
    recall = data.groupby(group_var).apply(fr.lag_crp)
    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp.pdf'))

    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **subj_kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_subject.pdf'))

    # crp by condition within
    recall = data.groupby(group_var).apply(fr.lag_crp, test_key='category',
                                           test=lambda x, y: x == y)
    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_within.pdf'))

    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **subj_kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_within_subject.pdf'))

    # crp by condition across
    recall = data.groupby(group_var).apply(fr.lag_crp, test_key='category',
                                           test=lambda x, y: x != y)
    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_across.pdf'))

    g = fr.plot_lag_crp(recall, hue=group_var, palette=palette, **subj_kwargs)
    g.add_legend()
    g.savefig(os.path.join(out_dir, 'crp_across_subject.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir')
    parser.add_argument('csv_file')
    parser.add_argument('--comp', '-c', help="Comparison data file")
    parser.add_argument('--group', '-g', help="Variable to group by.")
    args = parser.parse_args()

    main(args.out_dir, args.csv_file, comp_csv=args.comp, group_var=args.group)
