"""Figures for visualizing behavior and fits."""

import os
from pkg_resources import resource_filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from psifr import fr


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resource_filename('dcmr', 'figures.mplstyle')
    plt.style.use(style_path)


def save_image(fig, fig_file):
    """Save an image with no whitespace."""
    ax = fig.axes[0]
    ax.set_axis_off()
    ax.margins(0, 0)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(fig_file, pad_inches=0, bbox_inches='tight')


def plot_fit(
    data,
    group_var,
    stat_name,
    f_stat,
    stat_kws,
    stat_var,
    cond_var,
    f_plot,
    plot_kws,
    out_dir,
    ext='pdf',
):
    """
    Plot fit for an analysis and save figures.

    Parameters
    ----------
    data : DataFrame
        Observed and simulated data.
    
    group_var : str
        Column to split by for comparison.

    stat_name : str
        Base of file names for figures.
    
    f_stat : callable
        Takes data as the first input and returns a DataFrame with statistics.
    
    stat_kws : dict
        Keyword arguments for f_stat.
    
    stat_var : str
        Statistic column to plot.
    
    cond_var : str
        Column of statistics to plot by hue.
    
    f_plot : callable
        Seaborn-style figure-level function.
    
    plot_kws : dict
        Keyword arguments for f_plot.
    
    out_dir : str
        Directory to save plots in.
    
    ext : str
        Extension of saved plots.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    palette = sns.color_palette('viridis', 2)

    # mean stat
    stat = data.groupby(group_var).apply(f_stat, **stat_kws).droplevel(1)
    g = f_plot(stat, hue=group_var, palette=palette, height=4, **plot_kws)
    g.savefig(os.path.join(out_dir, f'{stat_name}.{ext}'))
    plt.close(g.fig)

    # subject stats
    g = f_plot(
        stat,
        hue=group_var,
        palette=palette,
        col='subject',
        col_wrap=6,
        height=3,
        **plot_kws,
    )
    g.savefig(os.path.join(out_dir, f'{stat_name}_subject.{ext}'))
    plt.close(g.fig)

    # comparison scatter plot
    groups = stat.index.unique()

    # by mean
    stat_index = stat.set_index(cond_var, append=True)[stat_var]
    m = stat_index.groupby([group_var, cond_var]).mean()
    comp = m.unstack(level=0).reset_index()
    if comp[cond_var].dtype == 'int64[pyarrow]':
        comp[cond_var] = comp[cond_var].astype(int)
    g = sns.relplot(
        kind='scatter',
        x=groups[0],
        y=groups[1],
        hue=cond_var,
        data=comp.reset_index(),
        height=4,
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp.{ext}'))
    plt.close(g.fig)

    # by subject
    stat_subj = stat.set_index([cond_var, 'subject'], append=True)[stat_var]
    comp = stat_subj.unstack(level=0).reset_index()
    if comp[cond_var].dtype == 'int64[pyarrow]':
        comp[cond_var] = comp[cond_var].astype(int)
    g = sns.relplot(
        kind='scatter',
        x=groups[0],
        y=groups[1],
        hue=cond_var,
        data=comp.reset_index(),
        height=4,
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.{ext}'))
    plt.close(g.fig)


def plot_fit_scatter(
    data, group_var, stat_name, f_stat, stat_kws, var_name, out_dir, ext='pdf'
):
    """
    Plot fit scatterplot for a scalar statistic.
    
    Parameters
    ----------
    data : DataFrame
        Observed and simulated data.
    
    group_var : str
        Column to split by for comparison.

    stat_name : str
        Base of file names for figures.
    
    f_stat : callable
        Takes data as the first input and returns a DataFrame with statistics.
    
    stat_kws : dict
        Keyword arguments for f_stat.
    
    var_name : str
        Statistic column to plot.
    
    out_dir : str
        Directory to save plots in.
    
    ext : str
        Extension of saved plots.
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    stat = data.groupby(group_var).apply(f_stat, **stat_kws).droplevel(1)
    groups = stat.index.unique()

    stat_index = stat.set_index('subject', append=True)[var_name]
    comp = stat_index.unstack(level=0)
    g = sns.relplot(
        kind='scatter', x=groups[0], y=groups[1], data=comp.reset_index(), height=4
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.{ext}'))
    plt.close(g.fig)


def plot_swarm_bar(
    data,
    x=None,
    y=None,
    hue=None,
    dark=None,
    light=None,
    ax=None,
    dodge=False,
    capsize=0.425,
    point_kind='swarm',
):
    """
    Make a bar plot with individual points and error bars.
    
    Deprecated. Use the swarmbar package instead.
    """
    if dark is None:
        dark = 'ch:rot=-.5, light=.7, dark=.3, gamma=.6'
    if light is None:
        light = 'ch:rot=-.5, light=.7, dark=.3, gamma=.2'

    if ax is None:
        ax = plt.gca()

    # plot individual points
    if point_kind == 'swarm':
        sns.swarmplot(
            data=data.reset_index(),
            x=x,
            y=y,
            hue=hue,
            palette=dark,
            size=4,
            linewidth=0.1,
            edgecolor='k',
            alpha=0.8,
            ax=ax,
            zorder=3,
            dodge=dodge,
        )
    elif point_kind == 'strip':
        sns.stripplot(
            data=data.reset_index(),
            x=x,
            y=y,
            hue=hue,
            palette=dark,
            size=4,
            linewidth=0.1,
            edgecolor='k',
            alpha=0.8,
            ax=ax,
            zorder=3,
            dodge=dodge,
        )
    else:
        raise ValueError(f'Invalid point plot kind: {point_kind}')

    # plot error bars for the mean
    sns.barplot(
        data=data.reset_index(),
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        dodge=dodge,
        color='k',
        palette=light,
        capsize=capsize,
        edgecolor='k',
        linewidth=0.75,
        err_kws={'color': 'k', 'linewidth': 0.8},
    )

    # remove overall xlabel and increase size of x-tick labels
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize='large')

    # fix ordering of plot elements
    plt.setp(ax.lines, zorder=100, linewidth=1.25, label=None)
    plt.setp(ax.collections, zorder=100, label=None)

    # delete legend (redundant with the x-tick labels)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
        if hue is not None:
            # refresh the legend to remove the swarm points
            ax.legend()


def remove_subject_variance(data, var_name, id_vars):
    """
    Remove subject variance from a variable for plotting.

    This approach is taken from Loftus & Masson 1994. After removing
    subject variance, only condition variance and interaction variance
    will remain; these are generally the components of interest in
    within-subject designs.
    """
    mean = data.groupby(id_vars)[var_name].transform('mean')
    deviation = mean - data[var_name].mean()
    normalized = data[var_name] - deviation
    return normalized


def plot_xval_comp(xval, aspect=None, height=None):
    """Plot cross-validation stat comparison."""
    xval['deviation'] = remove_subject_variance(xval, 'logl_test_list', 'subject')
    xval['mean_deviation'] = xval.groupby('model')['deviation'].transform('mean')
    xval_sorted = xval.sort_values('mean_deviation').reset_index()
    g = sns.catplot(
        data=xval_sorted,
        x='deviation',
        y='model',
        kind='point',
        linestyle='none',
        aspect=aspect,
        height=height,
    )
    g.set(xlabel='Cross-validated log likelihood', ylabel='Model variant')
    return g


def plot_dist_rank_asym(data, dark, light, ax=None):
    """Plot distance rank window statistic asymmetry."""
    if ax is None:
        ax = plt.gca()

    # plot individual points for all models
    plot_swarm_bar(
        data,
        x='rank',
        y='source',
        hue='source',
        dark=dark,
        light=light,
        point_kind='strip',
        ax=ax,
    )
    ax.tick_params(axis='x', labelsize='small')
    ax.tick_params(axis='y', labelsize='large')
    ax.axvline(0, *ax.get_ylim(), linewidth=1, color='k')
    ax.set(
        ylabel="",
        xlabel="Semantic clustering asymmetry",
    )

    # add annotation to help with interpretation
    prop = dict(
        rotation='horizontal',
        xycoords='axes fraction',
        verticalalignment='center',
        fontsize='large',
    )
    yoffset = -0.2
    xoffset = 0.1
    ax.annotate(
        'Rel. to prev. item',
        xy=(0, yoffset),
        horizontalalignment='left',
        **prop,
    )
    ax.annotate(
        'Rel. to next item',
        xy=(1, yoffset),
        horizontalalignment='right',
        **prop,
    )
    ax.annotate(
        '',
        xy=(0.5 - xoffset, yoffset),
        xytext=(0.5 + xoffset, yoffset),
        arrowprops=dict(arrowstyle="<|-|>", facecolor='k'),
        xycoords='axes fraction',
    )
    ax.xaxis.set_label_coords(0.5, -0.28)

    # plot mean of the data as a line for comparison
    m = data.loc['Data', 'rank'].mean()
    ax.axline([m, 0], slope=np.inf, linestyle='--', linewidth=0.5, color='k')
    return ax


def plot_trial_evidence(evidence, subject):
    """Plot evidence for each category by trial."""
    ml = pd.melt(
        fr.filter_data(evidence, subjects=subject),
        id_vars=['subject', 'list', 'position', 'trial_type'],
        value_vars=['cel', 'loc', 'obj'],
        var_name='category',
        value_name='evidence',
    )
    g = sns.relplot(
        data=ml,
        kind='line',
        x='position',
        y='evidence',
        hue='category',
        col='list',
        col_wrap=6,
        height=3,
    )
    g.set_xlabels('Serial position')
    g.set_ylabels('Evidence')
    return g


def plot_block_pos_evidence(mean_evidence):
    """Plot mean evidence by block position."""
    ml = pd.melt(
        mean_evidence.reset_index(),
        id_vars=['subject', 'block_pos'],
        value_vars=['curr', 'prev', 'base'],
        var_name='category',
        value_name='evidence',
    )
    g = sns.relplot(
        data=ml,
        x='block_pos',
        y='evidence',
        hue='category',
        col='subject',
        col_wrap=5,
        kind='line',
        height=4,
    )
    g.set_xlabels('Block position')
    g.set_ylabels('Evidence')
    return g


def plot_mean_block_pos_evidence(mean_evidence):
    """Plot group mean evidence by block position."""
    ml = pd.melt(
        mean_evidence.reset_index(),
        id_vars=['subject', 'block_pos'],
        value_vars=['curr', 'prev', 'base'],
        var_name='category',
        value_name='evidence',
    )
    ml['Category'] = ml['category'].map(
        {'curr': 'Current', 'prev': 'Previous', 'base': 'Baseline'}
    )
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    sns.lineplot(
        data=ml.query('category == "curr" and block_pos <= 3'),
        x='block_pos',
        y='evidence',
        hue='Category',
        palette=['C2'],
        errorbar=None,
        ax=ax[0],
    )
    sns.lineplot(
        data=ml.query('category != "curr" and block_pos <= 3'),
        x='block_pos',
        y='evidence',
        hue='Category',
        palette=['C0', 'C1'],
        errorbar=None,
        ax=ax[1],
    )
    ax[0].set(ylabel='Evidence', xlabel='Block position', xticks=[1, 2, 3])
    ax[1].set(ylabel='', xlabel='Block position', xticks=[1, 2, 3], xlim=[0.75, 3.25])
    return fig, ax


def plot_slope_evidence(slope):
    """Plot evidence slope by category type."""
    ml_slopes = pd.melt(
        slope,
        value_vars=['curr', 'prev', 'base'],
        var_name='category',
        value_name='slope',
    )

    fig, ax = plt.subplots(figsize=(5, 5))
    sns.swarmplot(
        data=ml_slopes,
        x='category',
        hue='category',
        y='slope',
        palette=['C2', 'C0', 'C1'],
        zorder=1,
        legend=False,
        ax=ax,
    )
    g = sns.pointplot(
        data=ml_slopes,
        x='category',
        y='slope',
        color='k',
        linestyle="none",
        zorder=2,
        ax=ax,
        capsize=0.4,
    )
    g.set_xticklabels(['Current', 'Previous', 'Baseline'], fontsize='large')
    g.set(xlabel='', ylabel='Evidence slope')
    x_lim = ax.get_xlim()
    ax.hlines(0, *x_lim, colors='k')
    ax.set(xlim=x_lim)
    return fig, ax
