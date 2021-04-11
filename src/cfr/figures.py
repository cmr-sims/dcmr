"""Figures for visualizing behavior and fits."""

import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import jinja2 as jn


def plot_fit(
    data, group_var, stat_name, f_stat, stat_kws, var_name, f_plot, plot_kws, out_dir,
    ext='pdf'
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
        stat, hue=group_var, palette=palette, col='subject', col_wrap=6, height=3,
        **plot_kws
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
        kind='scatter', x=groups[0], y=groups[1], hue=cond_name,
        data=comp.reset_index(), height=4
    )
    g.axes[0, 0].plot([0, 1], [0, 1], '-k')
    g.savefig(os.path.join(out_dir, f'{stat_name}_comp.{ext}'))
    plt.close(g.fig)

    # by subject
    comp = stat.unstack(level=0).droplevel(axis=1, level=0)
    g = sns.relplot(
        kind='scatter', x=groups[0], y=groups[1], hue=cond_name,
        data=comp.reset_index(), height=4
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


def render_fit_html(fit_dir, curves, points, ext='svg'):
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
            'subj': os.path.join(fit_dir, 'figs', f'{curve}_comp_subject.{ext}')
        }
        d_curve[curve] = entry

    # define points to include
    d_point = {}
    for point, analyses in points.items():
        entry = {
            analysis: os.path.join(
                fit_dir, 'figs', f'{analysis}_comp_subject.{ext}'
            ) for analysis in analyses
        }
        d_point[point] = entry

    # tables
    fit = pd.read_csv(os.path.join(fit_dir, 'fit.csv'))
    opt = {'float_format': '%.2f'}
    table_opt = {'Summary': {**opt}, 'Parameters': {'index': False, **opt}}

    # subject parameters and stats
    table = fit.copy().drop(columns=['rep'])
    table = table.astype({'subject': int, 'n': int, 'k': int})

    # summary statistics
    summary = (
        table.drop(columns=['subject', 'n', 'k'])
        .agg(['mean', 'sem', 'min', 'max'])
    )
    tables = {'Summary': summary, 'Parameters': table}

    # write html
    page = template.render(
        model=model, curves=d_curve, points=d_point, tables=tables, table_opt=table_opt
    )
    html_file = os.path.join(fit_dir, 'report.html')
    with open(html_file, 'w') as f:
        f.write(page)

    # copy css
    css = env.get_template('bootstrap.min.css')
    css_file = os.path.join(fit_dir, 'bootstrap.min.css')
    with open(css_file, 'w') as f:
        f.write(css.render())
    css = env.get_template('report.css')
    css_file = os.path.join(fit_dir, 'report.css')
    with open(css_file, 'w') as f:
        f.write(css.render())
