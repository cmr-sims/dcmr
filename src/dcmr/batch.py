"""Utilities for running commands in batches."""

import numpy as np
import click
from dcmr import cli
from dcmr import framework


def split_opt(opt):
    """Split an option list."""
    if opt is not None:
        opt_list = opt.split(',')
    else:
        opt_list = []
    return opt_list


def expand_opt_lists(*opt_lists):
    """"""
    max_n = np.max([len(ol) for ol in opt_lists])
    expand_lists = []
    for ol in opt_lists:
        if not ol:
            el = [None] * max_n
        elif len(ol) == 1:
            el = ol * max_n
        else:
            el = ol
        expand_lists.append(el)
    return tuple(expand_lists)


def expand_variants(fcf_features, ff_features, sublayer_param, fixed_param, free_param, dependent_param):
    """Expand variant lists to make full specifications."""
    fcf_list = split_opt(fcf_features)
    ff_list = split_opt(ff_features)
    sub_list = split_opt(sublayer_param)
    fix_list = split_opt(fixed_param)
    free_list = split_opt(free_param)
    dep_list = split_opt(dependent_param)

    fcf_list, ff_list, sub_list, fix_list, free_list, dep_list = expand_opt_lists(
        fcf_list, ff_list, sub_list, fix_list, free_list, dep_list
    )
    return fcf_list, ff_list, sub_list, fix_list, free_list, dep_list


def command_fit_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    subpar,
    fixed,
    free,
    dependent,
    n_reps=10,
    n_jobs=48,
    tol=0.00001,
    n_sim_reps=50,
):
    """Generate command line arguments for fitting CMR."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    inputs = f'{data_file} {patterns_file}'
    res_name = framework.generate_model_name(
        fcf_features, ff_features, intercept, sublayers, scaling, subpar, fixed, free, dependent
    )
    opts = f'-t {tol:.6f} -n {n_reps} -j {n_jobs} -r {n_sim_reps}'

    if sublayers:
        opts = f'--sublayers {opts}'
    else:
        opts = f'--no-sublayers {opts}'

    if scaling:
        opts = f'--scaling {opts}'
    else:
        opts = f'--no-scaling {opts}'

    if intercept:
        opts = f'--intercept {opts}'
    else:
        opts = f'--no-intercept {opts}'

    if subpar:
        opts += f' -p {subpar}'
    if fixed:
        opts += f' -f {fixed}'
    if free:
        opts += f' -e {free}'
    if dependent:
        opts += f' -a {dependent}'
    full_dir = study_dir / study / 'fits' / fit / res_name

    print(f'dcmr-fit {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("fcf_features")
@click.argument("ff_features")
@cli.model_options
@cli.fit_options
@cli.sim_options
def plan_fit_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
    **kwargs,
):
    """Print command lines for fitting multiple models."""
    fcf_list, ff_list, sub_list, fix_list, free_list, dep_list = expand_variants(
        fcf_features, ff_features, sublayer_param, fixed_param, free_param, dependent_param
    )
    for fcf, ff, sub, fix, free, dep in zip(fcf_list, ff_list, sub_list, fix_list, free_list, dep_list):
        command_fit_cmr(
            study,
            fit,
            fcf,
            ff,
            intercept,
            sublayers,
            scaling,
            sub,
            fix,
            free,
            dep,
            **kwargs,
        )


def command_xval_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    subpar,
    fixed,
    free,
    dependent,
    n_folds=None,
    fold_key=None,
    n_reps=10,
    n_jobs=48,
    tol=0.00001,
):
    """Generate command line arguments for fitting CMR."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    inputs = f'{data_file} {patterns_file}'
    res_name = framework.generate_model_name(
        fcf_features, ff_features, intercept, sublayers, scaling, subpar, fixed, free, dependent
    )
    opts = f'-t {tol:.6f} -n {n_reps} -j {n_jobs}'
    if n_folds is not None:
        opts += f' -d {n_folds}'
    if fold_key is not None:
        opts += f' -k {fold_key}'

    if sublayers:
        opts = f'--sublayers {opts}'
    else:
        opts = f'--no-sublayers {opts}'
    
    if scaling:
        opts = f'--scaling {opts}'
    else:
        opts = f'--no-scaling {opts}'

    if intercept:
        opts = f'--intercept {opts}'
    else:
        opts = f'--no-intercept {opts}'

    if subpar:
        opts += f' -p {subpar}'
    if fixed:
        opts += f' -f {fixed}'
    if free:
        opts += f' -e {free}'
    if dependent:
        opts += f' -a {dependent}'
    full_dir = study_dir / study / 'fits' / fit / res_name

    print(f'dcmr-xval {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("fcf_features")
@click.argument("ff_features")
@cli.model_options
@cli.fit_options
@cli.xval_options
def plan_xval_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    intercept,
    sublayers,
    scaling,
    sublayer_param,
    fixed_param,
    free_param,
    dependent_param,
    **kwargs,
):
    """Print command lines for fitting multiple models."""
    fcf_list, ff_list, sub_list, fix_list, free_list, dep_list = expand_variants(
        fcf_features, ff_features, sublayer_param, fixed_param, free_param, dependent_param
    )
    for fcf, ff, sub, fix, free, dep in zip(fcf_list, ff_list, sub_list, fix_list, free_list, dep_list):
        command_xval_cmr(
            study,
            fit,
            fcf,
            ff,
            intercept,
            sublayers,
            scaling,
            sub,
            fix,
            free,
            dep,
            **kwargs,
        )


def command_sim_cmr(study, fit, model, n_rep=1):
    """Generate command line arguments for simulating CMR."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    fit_dir = study_dir / study / 'fits' / fit / model
    if not fit_dir.exists():
        raise IOError(f'Fit directory does not exist: {fit_dir}')
    print(f'dcmr-sim {data_file} {patterns_file} {fit_dir} -r {n_rep}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("models")
@cli.sim_options
def plan_sim_cmr(study, fit, models, n_sim_reps):
    """Print command lines for simulating multiple models."""
    for model in models.split(","):
        command_sim_cmr(study, fit, model, n_sim_reps)


def command_plot_fit(study, fit, model, ext="svg"):
    """Generate command line arguments for plotting CMR simulations."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    fit_dir = study_dir / study / 'fits' / fit / model
    if not fit_dir.exists():
        raise IOError(f'Fit directory does not exist: {fit_dir}')
    print(f'dcmr-plot-fit -e {ext} {data_file} {patterns_file} {fit_dir}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("models")
@click.option("--ext", "-e", default="svg", help="figure file type (default: svg)")
def plan_plot_fit(study, fit, models, **kwargs):
    """Print command lines for plotting fit for multiple models."""
    for model in models.split(","):
        command_plot_fit(study, fit, model, **kwargs)


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("model_name")
@click.argument("flags")
def plan_compose_fit(study, fit, model_name, flags):
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    fit_dir = study_dir / study / 'fits' / fit / model_name

    if study == 'cfr':
        print(f'dcmr-fit-cfr-disrupt {data_file} {patterns_file} {fit_dir} {flags}')
