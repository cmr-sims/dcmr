"""Reports for summarizing behavior and model fit."""

import os
import logging
import jinja2 as jn
import numpy as np
from scipy.spatial import distance
import pandas as pd
import matplotlib
import seaborn as sns

matplotlib.use('Agg')
from psifr import fr
from dcmr import task
from dcmr import framework
from dcmr import figures


def render_fit_html(fit, report_dir, curves, points, grids=None, snapshots=None, ext='svg'):
    """
    Create an HTML report with figures and parameter tables.

    After running, open {report_dir}/report.html in a browser to view the
    report.

    Parameters
    ----------
    fit : pandas.DataFrame
        Fit results.

    report_dir : str
        Path to report directory.
    
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
    
    snapshots : list of str
        Model snapshots to include. Files named 
        snapshot_{snapshot}.{ext} will be included.
    
    ext : str
        File extension to expect for plot files. SVG files are 
        recommended for reports.
    """
    env = jn.Environment(
        loader=jn.PackageLoader('dcmr', 'templates'),
        autoescape=jn.select_autoescape(['html']),
    )
    template = env.get_template('report.html')
    model = os.path.basename(os.path.abspath(report_dir))

    # define curve plots to include
    d_curve = {}
    for curve in curves:
        entry = {
            'mean': os.path.join(report_dir, 'figs', f'{curve}.{ext}'),
            'comp': os.path.join(report_dir, 'figs', f'{curve}_comp.{ext}'),
            'subj': os.path.join(report_dir, 'figs', f'{curve}_comp_subject.{ext}'),
        }
        d_curve[curve] = entry

    # define points to include
    d_point = {}
    for point, analyses in points.items():
        entry = {
            analysis: os.path.join(report_dir, 'figs', f'{analysis}_comp_subject.{ext}')
            for analysis in analyses
        }
        d_point[point] = entry

    # define subject curves to include
    if grids is not None:
        d_grid = {
            grid: os.path.join(report_dir, 'figs', f'{grid}_subject.{ext}')
            for grid in grids
        }
    else:
        d_grid = None
    
    # snapshots
    if snapshots is not None:
        d_snapshot = {
            snapshot: (
                os.path.join(report_dir, 'figs', f'snapshot_{snapshot}.{ext}'),
                os.path.join(report_dir, 'figs', f'support_{snapshot}.{ext}'),
            )
            for snapshot in snapshots
        }
    else:
        d_snapshot = None

    # tables
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
        snapshots=d_snapshot,
        tables=tables,
        table_opt=table_opt,
    )
    html_file = os.path.join(report_dir, 'report.html')
    with open(html_file, 'w') as f:
        f.write(page)

    # copy css
    css = env.get_template('bootstrap.min.css')
    os.makedirs(os.path.join(report_dir, '.css'), exist_ok=True)
    css_file = os.path.join(report_dir, '.css', 'bootstrap.min.css')
    with open(css_file, 'w') as f:
        f.write(css.render())
    css = env.get_template('report.css')
    css_file = os.path.join(report_dir, '.css', 'report.css')
    with open(css_file, 'w') as f:
        f.write(css.render())


def within(x, y):
    return x == y


def across(x, y):
    return x != y


def plot_fit(
    data, 
    sim, 
    group_param,
    subj_param,
    param_def,
    patterns,
    fit,
    report_dir, 
    ext='svg', 
    study_keys=None,
    snapshot=True,
    category=None,
    similarity=None,
    block=None,
    data_filter=None,
):
    """Make a report with fit information."""
    # information about the data
    if category is None:
        asymfr = 'list_type' in data.columns and 'toronto' in data['list_type'].unique()
        category = 'category' in data.columns and not asymfr
    if block is None:
        block = 'n_block' in data.columns and 'block' in data.columns
    if similarity is None:
        similarity = 'use' in patterns['vector']
    if study_keys is None:
        study_keys = task.get_study_keys(data)

    # filter the data
    if data_filter is not None:
        logging.info(f'Applying filter to data: {data_filter}')
        data = data.query(data_filter).copy()
        sim = sim.query(data_filter).copy()

    # prep category tests
    if category:
        test = {'within': within, 'across': across}

    # prep semantic similarity
    if similarity:
        distances = distance.squareform(
            distance.pdist(patterns['vector']['use'], 'correlation')
        )
        edges = np.percentile(distance.squareform(distances), np.linspace(1, 100, 11))
    data['item_index'] = fr.pool_index(data['item'], patterns['items'])
    sim['item_index'] = fr.pool_index(sim['item'], patterns['items'])

    # merge and concatenate for analysis
    data_merge = task.merge_free_recall(data)
    sim_merge = task.merge_free_recall(sim)
    full = pd.concat((data_merge, sim_merge), axis=0, keys=['Data', 'Model'])
    full.index.rename(['source', 'trial'], inplace=True)

    # make plots
    fig_dir = os.path.join(report_dir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)
    kwargs = {'ext': ext}

    # snapshots
    if snapshot:
        figures.plot_model_snapshots(
            data, 
            group_param, 
            subj_param, 
            param_def, 
            patterns, 
            study_keys, 
            fig_dir, 
            **kwargs,
        )

    # scalar stats
    logging.info('Plotting fits to individual scalar statistics.')

    # distance rank
    if similarity:
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
            for cond in ['within', 'across']:
                figures.plot_fit_scatter(
                    full,
                    'source',
                    f'use_rank_{cond}',
                    fr.distance_rank,
                    {
                        'index_key': 'item_index',
                        'distances': distances,
                        'test_key': 'category',
                        'test': test[cond],
                    },
                    'rank',
                    fig_dir,
                    **kwargs,
                )
    
    # lag rank
    figures.plot_fit_scatter(
        full, 'source', 'lag_rank', fr.lag_rank, {}, 'rank', fig_dir, **kwargs
    )
    if category:
        for cond in ['within', 'across']:
            figures.plot_fit_scatter(
                full,
                'source',
                f'lag_rank_{cond}',
                fr.lag_rank,
                {'test_key': 'category', 'test': test[cond]},
                'rank',
                fig_dir,
                **kwargs,
            )
    
    # category crp
    if category:
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
        if block:
            figures.plot_fit_scatter(
                full,
                'source',
                'cat_crp_remote',
                fr.category_crp,
                {
                    'category_key': 'category', 
                    'test_key': 'block', 
                    'test': test['across']
                },
                'prob',
                fig_dir,
                **kwargs,
            )

    # curves
    logging.info('Plotting fits to curves.')

    # compound lag CRP
    figures.plot_fit(
        full,
        'source',
        'lag_crp_compound',
        fr.lag_crp_compound,
        {},
        'prob',
        ['current', 'previous'],
        figures.plot_lag_crp_compound,
        {},
        fig_dir,
        **kwargs,
    )
    if category:
        for cond in ['within', 'across']:
            figures.plot_fit(
                full,
                'source',
                f'lag_crp_compound_{cond}',
                fr.lag_crp_compound,
                {'test_key': 'category', 'test': test[cond]},
                'prob',
                ['current', 'previous'],
                figures.plot_lag_crp_compound,
                {},
                fig_dir,
                **kwargs,
            )

    if block:
        # block SPC
        figures.plot_fit(
            full,
            'source',
            'block_spc',
            task.block_spc,
            {'block_key': 'block', 'n_block_key': 'n_block'},
            'recall',
            ['block', 'n_block'],
            fr.plot_spc,
            {'input_key': 'block'},
            fig_dir,
            **kwargs,
        )

        # block lag CRP
        figures.plot_fit(
            full,
            'source',
            'block_lag_crp',
            task.block_lag_crp,
            {'block_key': 'block', 'n_block_key': 'n_block'},
            'prob',
            ['lag', 'n_block'],
            figures.plot_block_lag_crp,
            {},
            fig_dir,
            **kwargs,
        )
        if category:
            for cond in ['within', 'across']:
                figures.plot_fit(
                    full,
                    'source',
                    f'block_lag_crp_{cond}',
                    task.block_lag_crp,
                    {
                        'block_key': 'block', 
                        'n_block_key': 'n_block', 
                        'test_key': 'category', 
                        'test': test[cond],
                    },
                    'prob',
                    ['lag', 'n_block'],
                    figures.plot_block_lag_crp,
                    {},
                    fig_dir,
                    **kwargs,
                )

    # input CRP
    figures.plot_fit(
        full,
        'source',
        'input_crp',
        fr.input_crp,
        {},
        'prob',
        ['previous', 'current'],
        fr.plot_input_crp,
        {},
        fig_dir,
        **kwargs,
    )

    if similarity:
        # distance CRP
        figures.plot_fit(
            full,
            'source',
            'use_crp',
            fr.distance_crp,
            {'index_key': 'item_index', 'distances': distances, 'edges': edges},
            'prob',
            'center',
            fr.plot_distance_crp,
            {'min_samples': None},
            fig_dir,
            **kwargs,
        )
        if category:
            for cond in ['within', 'across']:
                figures.plot_fit(
                    full,
                    'source',
                    f'use_crp_{cond}',
                    fr.distance_crp,
                    {
                        'index_key': 'item_index',
                        'distances': distances,
                        'edges': edges,
                        'test_key': 'category',
                        'test': test[cond],
                    },
                    'prob',
                    'center',
                    fr.plot_distance_crp,
                    {'min_samples': None},
                    fig_dir,
                    **kwargs,
                )

        # shifted distance CRP
        figures.plot_fit(
            full,
            'source',
            'use_rank_shifted',
            fr.distance_rank_shifted,
            {'index_key': 'item_index', 'distances': distances, 'max_shift': 3},
            'rank',
            'shift',
            figures.plot_distance_rank_shifted,
            {},
            fig_dir,
            **kwargs,
        )
        if category:
            for cond in ['within', 'across']:
                figures.plot_fit(
                    full,
                    'source',
                    f'use_rank_shifted_{cond}',
                    fr.distance_rank_shifted,
                    {
                        'index_key': 'item_index', 
                        'distances': distances, 
                        'max_shift': 3,
                        'test_key': 'category',
                        'test': test[cond],
                    },
                    'rank',
                    'shift',
                    figures.plot_distance_rank_shifted,
                    {},
                    fig_dir,
                    **kwargs,
                )
    
    # SPC
    figures.plot_fit(
        full, 'source', 'spc', fr.spc, {}, 'recall', 'input', fr.plot_spc, {}, fig_dir, **kwargs
    )

    # PFR
    figures.plot_fit(
        full,
        'source',
        'pfr',
        lambda x: fr.pnr(x).query('output <= 3'),
        {},
        'prob',
        ['input', 'output'],
        fr.plot_spc,
        {},
        fig_dir,
        **kwargs,
    )

    # lag CRP
    figures.plot_fit(
        full,
        'source',
        'lag_crp',
        fr.lag_crp,
        {},
        'prob',
        'lag',
        fr.plot_lag_crp,
        {'max_lag': None},
        fig_dir,
        **kwargs,
    )
    if category:
        for cond in ['within', 'across']:
            figures.plot_fit(
                full,
                'source',
                f'lag_crp_{cond}',
                fr.lag_crp,
                {'test_key': 'category', 'test': test[cond]},
                'prob',
                'lag',
                fr.plot_lag_crp,
                {'max_lag': None},
                fig_dir,
                **kwargs,
            )

    # parameter pair plot
    g = sns.pairplot(fit[list(param_def.free.keys())])
    g.savefig(os.path.join(fig_dir, f'parameters_subject.{ext}'))

    # report
    curves = ['spc', 'pfr', 'input_crp', 'lag_crp']
    points = {'lag_rank': ['lag_rank']}
    if category:
        curves.extend(['lag_crp_within', 'lag_crp_across'])
        points['lag_rank'].extend(['lag_rank_within', 'lag_rank_across'])
        points['cat_crp'] = ['cat_crp']
        if block:
            points['cat_crp'].append('cat_crp_remote')
    curves.append('lag_crp_compound')
    if category:
        curves.extend(['lag_crp_compound_within', 'lag_crp_compound_across'])
    if block:
        curves.append('block_spc')
        curves.append('block_lag_crp')
        if category:
            curves.extend(['block_lag_crp_within', 'block_lag_crp_across'])
    if similarity:
        curves.append('use_crp')
        points['use_rank'] = ['use_rank']
        if category:
            curves.extend(['use_crp_within', 'use_crp_across'])
            points['use_rank'].extend(['use_rank_within', 'use_rank_across'])
        curves.append('use_rank_shifted')
        if category:
            curves.extend(['use_rank_shifted_within', 'use_rank_shifted_across'])
    grids = curves.copy() + ['parameters']
    if snapshot:
        snapshots = [f'sub-{subj}' for subj in subj_param.keys()]
    else:
        snapshots = None
    wd = os.getcwd()
    os.chdir(report_dir)
    render_fit_html(fit, '.', curves, points, grids, snapshots)
    os.chdir(wd)


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
        'B_disrupt': r'\beta_{\mathrm{disrupt}}',
        'B_start0': r'\beta_{\mathrm{start},\mathrm{0 s}}',
        'B_start1': r'\beta_{\mathrm{start},\mathrm{2.5 s}}',
        'B_start2': r'\beta_{\mathrm{start},\mathrm{7.5 s}}',
        'X10': r'\theta_{s,\mathrm{0 s}}',
        'X11': r'\theta_{s,\mathrm{2.5 s}}',
        'X12': r'\theta_{s,\mathrm{7.5 s}}',
        'X20': r'\theta_{r,\mathrm{0 s}}',
        'X21': r'\theta_{r,\mathrm{2.5 s}}',
        'X22': r'\theta_{r,\mathrm{7.5 s}}',
        'Lfc_loc_raw': r'L_{FC,I}',
        'Lfc_cat_raw': r'L_{FC,C}',
        'Lfc_use_raw': r'L_{FC,D}',
        'Lcf_loc_raw': r'L_{CF,I}',
        'Lcf_cat_raw': r'L_{CF,C}',
        'Lcf_use_raw': r'L_{CF,D}',
        'B_distract_raw_loc': r'\beta_{\mathrm{distract},I}',
        'B_distract_raw_cat': r'\beta_{\mathrm{distract},C}',
        'B_distract_raw_use': r'\beta_{\mathrm{distract},D}',
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
