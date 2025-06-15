"""Reports for summarizing behavior and model fit."""

import os
import logging
import jinja2 as jn
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib
import seaborn as sns
import click

matplotlib.use('Agg')
from psifr import fr
from cymr import cmr
from dcmr import task
from dcmr import framework
from dcmr import figures


def render_fit_html(fit_dir, curves, points, grids=None, ext='svg'):
    """
    Create an HTML report with figures and parameter tables.

    After running, open {fit_dir}/report.html in a browser to view the
    report.

    Parameters
    ----------
    fit_dir : str
        Path to directory with fit results.
    
    curves : list of str
        Names of curves to include in the report. Files named 
        {curve}.{ext}, {curve}_comp.{ext}, and {curve}_comp_subject.{ext}
        will be included.
    
    points : dict of (str: list of str)
        Groups of analyses to include. Files named 
        {analysis}_comp_subject.{ext} will be included.
    
    grids : list of str
        Plot grids to include. Files named {grid}_subject.{ext} will be
        included.
    
    ext : str
        File extension to expect for plot files. SVG files are 
        recommended for reports.
    """
    env = jn.Environment(
        loader=jn.PackageLoader('dcmr', 'templates'),
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


@click.command()
@click.argument("data_file")
@click.argument("patterns_file")
@click.argument("fit_dir")
@click.option("--ext", "-e", default="svg", help="figure file type (default: svg)")
def plot_fit(data_file, patterns_file, fit_dir, ext):
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
    category = 'category' in sim.columns

    # prep semantic similarity
    logging.info(f'Loading network patterns from {patterns_file}.')
    patterns = cmr.load_patterns(patterns_file)
    distances = distance.squareform(
        distance.pdist(patterns['vector']['use'], 'correlation')
    )
    edges = np.linspace(0.05, 0.95, 10)
    data['item_index'] = fr.pool_index(data['item'], patterns['items'])

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
        {'index_key': 'item_index', 'distances': distances},
        'rank',
        fig_dir,
        **kwargs,
    )
    if category:
        figures.plot_fit_scatter(
            full,
            'source',
            'use_rank_within',
            fr.distance_rank,
            {
                'index_key': 'item_index',
                'distances': distances,
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
                'distances': distances,
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
    if category:
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
        {'index_key': 'item_index', 'distances': distances, 'edges': edges},
        'prob',
        'center',
        fr.plot_distance_crp,
        {'min_samples': 10},
        fig_dir,
        **kwargs,
    )
    if category:
        figures.plot_fit(
            full,
            'source',
            'use_crp_within',
            fr.distance_crp,
            {
                'index_key': 'item_index',
                'distances': distances,
                'edges': edges,
                'test_key': 'category',
                'test': lambda x, y: x == y,
            },
            'prob',
            'center',
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
                'distances': distances,
                'edges': edges,
                'test_key': 'category',
                'test': lambda x, y: x != y,
            },
            'prob',
            'center',
            fr.plot_distance_crp,
            {'min_samples': 10},
            fig_dir,
            **kwargs,
        )
    figures.plot_fit(
        full, 'source', 'spc', fr.spc, {}, 'recall', 'input', fr.plot_spc, {}, fig_dir, **kwargs
    )
    figures.plot_fit(
        full,
        'source',
        'pfr',
        lambda x: fr.pnr(x).query('output == 1'),
        {},
        'prob',
        'input',
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
        'lag',
        fr.plot_lag_crp,
        {},
        fig_dir,
        **kwargs,
    )
    if category:
        figures.plot_fit(
            full,
            'source',
            'lag_crp_within',
            fr.lag_crp,
            {'test_key': 'category', 'test': lambda x, y: x == y},
            'prob',
            'lag',
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
            'lag',
            fr.plot_lag_crp,
            {},
            fig_dir,
            **kwargs,
        )

    # parameter pair plot
    fit = pd.read_csv(os.path.join(fit_dir, 'fit.csv'))
    param_def = cmr.read_config(os.path.join(fit_dir, 'parameters.json'))
    g = sns.pairplot(fit[list(param_def.free.keys())])
    g.savefig(os.path.join(fig_dir, f'parameters_subject.{ext}'))

    # report
    if category:
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
        grids = curves.copy() + ['parameters']
    else:
        curves = ['spc', 'pfr', 'lag_crp', 'use_crp']
        points = {'lag_rank': ['lag_rank'], 'use_rank': ['use_rank']}
        grids = curves.copy() + ['parameters']
    os.chdir(fit_dir)
    render_fit_html('.', curves, points, grids)


def get_param_latex():
    """Get the mapping from parameter names to LaTeX form."""
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
    """
    Create a summary table to compare models.

    Parameters
    ----------
    fit_dir : str
        Path to main directory with model fitting results. Individual
        model fits should be in subdirectories.

    models : list of str
        Model identifiers to include. Must match model directory names
        in fit_dir.
    
    model_names : list of str
        Model names to use in the table.
    
    param_map : dict of (str: str)
        Mapping of parameter names to display names, which may include
        LaTeX math code.
    
    model_comp : str
        Method for comparing models. May be "xval" or "aic".
    
    Returns
    -------
    reordered : pandas.DataFrame
        Data frame with the table, ordered to match the order of keys
        in param_map.
    """
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
    else:
        stats = []
        fields = free_param
        mean_only = []

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
    return reordered, table
