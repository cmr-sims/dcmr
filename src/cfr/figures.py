"""Figures for visualizing behavior and fits."""

import os
import seaborn as sns


def plot_fit(data, group_var, stat_name, f_stat, stat_kws, f_plot, plot_kws,
             out_dir):
    """Plot fit for an analysis and save figures."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    palette = sns.color_palette('viridis', 2)

    # mean stat
    stat = data.groupby(group_var).apply(f_stat, **stat_kws)
    g = f_plot(stat, hue=group_var, palette=palette, height=4, **plot_kws)
    g.savefig(os.path.join(out_dir, f'{stat_name}.pdf'))

    # subject stats
    g = f_plot(stat, hue=group_var, palette=palette, col='subject',
               col_wrap=6, height=3, **plot_kws)
    g.savefig(os.path.join(out_dir, f'{stat_name}_subject.pdf'))

    # comparison scatter plot
    groups = stat.index.levels[0]
    var_name = stat.index.names[2]

    # by mean
    if 'prob' in stat:
        stat = stat.loc[:, ['prob']].copy()
    m = stat.groupby([group_var, var_name]).mean()
    comp = m.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(kind='scatter', x=groups[0], y=groups[1], hue=var_name,
                    data=comp.reset_index(), height=4)
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp.pdf'))

    # by subject
    comp = stat.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(kind='scatter', x=groups[0], y=groups[1], hue=var_name,
                    data=comp.reset_index(), height=4)
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.pdf'))
