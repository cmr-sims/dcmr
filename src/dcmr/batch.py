"""Utilities for running commands in batches."""

import sys
import itertools
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
@cli.compose_options
@cli.fit_options
@cli.sim_options
@cli.filter_options
def plan_compose_fit(
    study,
    fit,
    semantics,
    cuing,
    intercept,
    free_t,
    disrupt_sublayers,
    special_sublayers,
    **_,
):
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    model_name = framework.compose_model_name(
        semantics,
        cuing,
        intercept,
        free_t,
        disrupt_sublayers,
        special_sublayers,
    )
    fit_dir = study_dir / study / 'fits' / fit / model_name
    flags = ' '.join(sys.argv[3:])

    if study == 'cfr':
        print(f'dcmr-fit-cfr-disrupt {data_file} {patterns_file} {fit_dir} {flags}')
    elif study == 'cdcatfr2':
        print(f'dcmr-fit-cdcatfr2 {data_file} {patterns_file} {fit_dir} {flags}')

def keywords_to_options(**kwargs):
    """Convert keyword arguments to command line options."""
    options = []
    for name, value in kwargs.items():
        if value is not None:
            name = name.replace('_', '-')
            if isinstance(value, bool):
                if value:
                    opt = f"--{name}"
                else:
                    opt = f"--no-{name}"
                options.append(opt)
            elif isinstance(value, tuple):
                for val in value:
                    options.append(f"--{name}={val}")
            else:
                options.append(f"--{name}={value}")
    return options


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("factors")
@click.argument("flags")
@cli.compose_options
def plan_compose_switchboard(
    study,
    fit,
    factors,
    flags,
    semantics,
    cuing,
    intercept,
    free_t,
    disrupt_sublayers,
    special_sublayers,
):
    """Print command lines for switchboard model evaluation."""
    expansions = {
        "sem": "semantics",
        "cue": "cuing",
        "dis": "disrupt_sublayers",
        "sub": "special_sublayers",
    }
    factors = [expansions[f] if f in expansions else f for f in factors.split("-")]
    d = {
        "intercept": [True, False],
        "free_t": [True, False],
        "semantics": ["context", "split", "item"],
        "cuing": ["integrative", "focused"],
        "disrupt_sublayers": [None, ("loc",), ("cat",), ("loc", "cat")],
        "special_sublayers": [None, ("list",), ("block",), ("list", "block")],
    }
    defaults = dict(
        semantics=semantics,
        cuing=cuing,
        intercept=intercept,
        free_t=free_t,
        disrupt_sublayers=disrupt_sublayers,
        special_sublayers=special_sublayers,
    )
    study_dir, data_file, patterns_file = framework.get_study_paths(study)

    levels = [d[key] for key in factors]
    product = itertools.product(*levels)
    for combination in product:
        # compose this variant with current levels and defaults
        features = dict(zip(factors, combination))
        all_features = defaults.copy()
        all_features.update(features)

        # screen out invalid variants
        if (
            all_features["semantics"] == "item"
            and all_features["disrupt_sublayers"] is not None
            and "cat" in all_features["disrupt_sublayers"]
        ):
            continue

        # construct the standard model name and output directory
        model_name = framework.compose_model_name(**all_features)
        fit_dir = study_dir / study / 'fits' / fit / model_name

        # print the command
        options = " ".join(keywords_to_options(**all_features))
        if study == 'cfr':
            print(f'dcmr-fit-cfr-disrupt {data_file} {patterns_file} {fit_dir} {flags} {options}')
        elif study == 'cdcatfr2':
            print(f'dcmr-fit-cdcatfr2 {data_file} {patterns_file} {fit_dir} {flags} {options}')
