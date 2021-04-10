"""Figures for visualizing behavior and fits."""

import os
import matplotlib.pyplot as plt
import seaborn as sns


def plot_fit(
    data, group_var, stat_name, f_stat, stat_kws, var_name, f_plot, plot_kws, out_dir
):
    """Plot fit for an analysis and save figures."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    palette = sns.color_palette('viridis', 2)

    # mean stat
    stat = data.groupby(group_var).apply(f_stat, **stat_kws)
    g = f_plot(stat, hue=group_var, palette=palette, height=4, **plot_kws)
    g.savefig(os.path.join(out_dir, f'{stat_name}.pdf'))
    plt.close(g.fig)

    # subject stats
    g = f_plot(
        stat, hue=group_var, palette=palette, col='subject', col_wrap=6, height=3,
        **plot_kws
    )
    g.savefig(os.path.join(out_dir, f'{stat_name}_subject.pdf'))
    plt.close(g.fig)

    # comparison scatter plot
    groups = stat.index.levels[0]
    cond_name = stat.index.names[-1]

    # by mean
    stat = stat[[var_name]]
    m = stat.groupby([group_var, cond_name]).mean()
    comp = m.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter', x=groups[0], y=groups[1], hue=cond_name,
        data=comp.reset_index(), height=4
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp.pdf'))
    plt.close(g.fig)

    # by subject
    comp = stat.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter', x=groups[0], y=groups[1], hue=cond_name,
        data=comp.reset_index(), height=4
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.pdf'))
    plt.close(g.fig)


def plot_fit_scatter(data, group_var, stat_name, f_stat, stat_kws, var_name, out_dir):
    """Plot fit scatterplot for a scalar statistic."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    stat = data.groupby(group_var).apply(f_stat, **stat_kws)
    groups = stat.index.levels[0]

    stat = stat[[var_name]]
    comp = stat.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter', x=groups[0], y=groups[1], data=comp.reset_index(), height=4
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.pdf'))
    plt.close(g.fig)
