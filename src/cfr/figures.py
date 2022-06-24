"""Figures for visualizing behavior and fits."""

import os
from pkg_resources import resource_filename
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jinja2 as jn
from cfr import framework


def set_style(style_path=None):
    """Set default plot style."""
    if style_path is None:
        style_path = resource_filename('cfr', 'figures.mplstyle')
    plt.style.use(style_path)


def plot_fit(
    data,
    group_var,
    stat_name,
    f_stat,
    stat_kws,
    var_name,
    f_plot,
    plot_kws,
    out_dir,
    ext='pdf',
):
    """Plot fit for an analysis and save figures."""
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    palette = sns.color_palette('viridis', 2)

    # mean stat
    stat = data.groupby(group_var).apply(f_stat, **stat_kws)
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
    groups = stat.index.levels[0]
    cond_name = stat.index.names[-1]

    # by mean
    stat = stat[[var_name]]
    m = stat.groupby([group_var, cond_name]).mean()
    comp = m.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter',
        x=groups[0],
        y=groups[1],
        hue=cond_name,
        data=comp.reset_index(),
        height=4,
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp.{ext}'))
    plt.close(g.fig)

    # by subject
    comp = stat.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter',
        x=groups[0],
        y=groups[1],
        hue=cond_name,
        data=comp.reset_index(),
        height=4,
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp_subject.{ext}'))
    plt.close(g.fig)


def plot_fit_scatter(
    data, group_var, stat_name, f_stat, stat_kws, var_name, out_dir, ext='pdf'
):
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
    """Make a bar plot with individual points and error bars."""
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
        errwidth=0.8,
        capsize=capsize,
        edgecolor='k',
        linewidth=0.75,
        errcolor='k',
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


def render_fit_html(fit_dir, curves, points, grids=None, ext='svg'):
    env = jn.Environment(
        loader=jn.PackageLoader('cfr', 'templates'),
        autoescape=jn.select_autoescape(['html']),
    )
    template = env.get_template('report.html')
    model = os.path.basename(os.path.abspath(fit_dir))

    # define curve plots to include
    d_curve = {}
    for curve in curves:
        entry = {
            'mean': os.path.join(fit_dir, 'figs', f'{curve}.{ext}'),
            'comp': os.path.join(fit_dir, 'figs', f'{curve}_comp.{ext}'),
            'subj': os.path.join(fit_dir, 'figs', f'{curve}_comp_subject.{ext}'),
        }
        d_curve[curve] = entry

    # define points to include
    d_point = {}
    for point, analyses in points.items():
        entry = {
            analysis: os.path.join(fit_dir, 'figs', f'{analysis}_comp_subject.{ext}')
            for analysis in analyses
        }
        d_point[point] = entry

    # define subject curves to include
    if grids is not None:
        d_grid = {
            grid: os.path.join(fit_dir, 'figs', f'{grid}_subject.{ext}')
            for grid in grids
        }
    else:
        d_grid = None

    # tables
    fit = pd.read_csv(os.path.join(fit_dir, 'fit.csv'))
    opt = {'float_format': '%.2f'}
    table_opt = {'Summary': {**opt}, 'Parameters': {'index': False, **opt}}

    # subject parameters and stats
    table = fit.copy().drop(columns=['rep'])
    table = table.astype({'subject': int, 'n': int, 'k': int})

    # summary statistics
    summary = table.drop(columns=['subject', 'n', 'k']).agg(
        ['mean', 'sem', 'min', 'max']
    )
    tables = {'Summary': summary, 'Parameters': table}

    # write html
    page = template.render(
        model=model,
        curves=d_curve,
        points=d_point,
        grids=d_grid,
        tables=tables,
        table_opt=table_opt,
    )
    html_file = os.path.join(fit_dir, 'report.html')
    with open(html_file, 'w') as f:
        f.write(page)

    # copy css
    css = env.get_template('bootstrap.min.css')
    os.makedirs(os.path.join(fit_dir, '.css'), exist_ok=True)
    css_file = os.path.join(fit_dir, '.css', 'bootstrap.min.css')
    with open(css_file, 'w') as f:
        f.write(css.render())
    css = env.get_template('report.css')
    css_file = os.path.join(fit_dir, '.css', 'report.css')
    with open(css_file, 'w') as f:
        f.write(css.render())


def get_param_latex():
    latex_names = {
        'Lfc': 'L_{FC}',
        'Lcf': 'L_{CF}',
        'Dff': 'D_{FF}',
        'P1': r'\phi_s',
        'P2': r'\phi_d',
        'B_enc': r'\beta_{\mathrm{enc}}',
        'B_enc_loc': r'\beta_{\mathrm{enc},I}',
        'B_enc_cat': r'\beta_{\mathrm{enc},C}',
        'B_enc_use': r'\beta_{\mathrm{enc},D}',
        'B_start': r'\beta_{\mathrm{start}}',
        'B_rec': r'\beta_{\mathrm{rec}}',
        'B_rec_loc': r'\beta_{\mathrm{rec},I}',
        'B_rec_cat': r'\beta_{\mathrm{rec},C}',
        'B_rec_use': r'\beta_{\mathrm{rec},D}',
        'X1': r'\theta_s',
        'X2': r'\theta_r',
        'w0': 'w_1',
        'w1': 'w_2',
        's0': 's_1',
        'k': 'k',
        'logl': r'\mathrm{log}(L)',
        'logl_test_list': r'\mathrm{log}(L)',
        'aic': r'\mathrm{AIC}',
        'waic': r'\mathrm{wAIC}',
    }
    math_format = {k: f'${v}$' for k, v in latex_names.items()}
    return math_format


def create_model_table(fit_dir, models, model_names, param_map=None, model_comp='xval'):
    """Create a summary table to compare models."""
    # get free parameters
    df = framework.read_model_specs(fit_dir, models, model_names)
    free_param = df.reset_index().query("kind == 'free'")['param'].unique()

    # get parameter values and likelihood
    res = framework.read_model_fits(fit_dir, models, model_names, param_map)
    if param_map is not None:
        free_param = [f for f in free_param if f not in param_map.keys()]

    if model_comp == 'aic':
        res = framework.model_comp_weights(res, stat='aic')
        stats = ['n', 'k', 'logl', 'aic', 'waic']
        fields = np.hstack((free_param, stats))
        mean_only = ['k']
    elif model_comp == 'xval':
        res = framework.read_model_xvals(fit_dir, models, model_names)
        stats = ['k', 'logl_test_list']
        fields = np.hstack((free_param, stats))
        mean_only = ['k']
    table = pd.DataFrame(index=fields, columns=model_names)

    # parameter means and sem
    model_stats = res.groupby('model').agg(['mean', 'sem'])
    for model in model_names:
        subset = df.reset_index().query(f"model == '{model}'")
        not_free_param = subset.query("kind != 'free'")['param'].unique().tolist()
        m = model_stats.loc[model]
        for field in fields:
            f = m[field]
            if np.isnan(f['mean']):
                table.loc[field, model] = '---'
            elif field in mean_only + not_free_param:
                table.loc[field, model] = f"{f['mean']:.0f}"
            else:
                table.loc[field, model] = f"{f['mean']:.2f} ({f['sem']:.2f})"

    # rename parameters to latex code
    latex_names = get_param_latex()
    order = [n for n in latex_names.keys() if n in table.index]
    reordered = table.reindex(order).rename(index=latex_names)
    return reordered
