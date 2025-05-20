"""Utilities for running commands in batches."""

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import click
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
        expand_lists.append(ol)
    return tuple(expand_lists)


def expand_variants(fcf_features, ff_features, sublayer_param, fixed_param):
    """Expand variant lists to make full specifications."""
    fcf_list = split_opt(fcf_features)
    ff_list = split_opt(ff_features)
    sub_list = split_opt(sublayer_param)
    fix_list = split_opt(fixed_param)

    fcf_list, ff_list, sub_list, fix_list = expand_opt_lists(
        fcf_list, ff_list, sub_list, fix_list
    )
    return fcf_list, ff_list, sub_list, fix_list


def command_fit_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    sublayers,
    subpar,
    fixed,
    n_reps=10,
    n_jobs=48,
    tol=0.00001,
    n_sim_reps=50,
):
    """Generate command line arguments for fitting CMR."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    inputs = f'{data_file} {patterns_file}'
    res_name = framework.generate_model_name(
        fcf_features, ff_features, sublayers, subpar, fixed
    )
    opts = f'-t {tol:.6f} -n {n_reps} -j {n_jobs} -r {n_sim_reps}'

    if sublayers:
        opts = f'--sublayers {opts}'
    else:
        opts = f'--no-sublayers {opts}'

    if subpar:
        opts += f' -p {subpar}'
    if fixed:
        opts += f' -f {fixed}'
    full_dir = study_dir / study / 'fits' / fit / res_name

    print(f'cfr-fit-cmr {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("fcf_features")
@click.argument("ff_features")
@click.option("--sublayers/--no-sublayers", default=False)
@click.option(
    "--sublayer-param",
    "-p",
    help="parameters free to vary between sublayers (e.g., B_enc-B_rec)",
)
@click.option(
    "--fixed-param",
    "-f",
    help="dash-separated list of values for fixed parameters (e.g., B_enc_cat=1)",
)
@click.option(
    "--n-reps",
    "-n",
    type=int,
    default=1,
    help="number of times to replicate the search",
)
@click.option(
    "--n-jobs", "-j", type=int, default=1, help="number of parallel jobs to use"
)
@click.option("--tol", "-t", type=float, default=0.00001, help="search tolerance")
@click.option(
    "--n-sim-reps",
    "-r",
    type=int,
    default=1,
    help="number of experiment replications to simulate",
)
def plan_fit_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    sublayers,
    sublayer_param,
    fixed_param,
    **kwargs,
):
    """Print command lines for fitting multiple models."""
    fcf_list, ff_list, sub_list, fix_list = expand_variants(
        fcf_features, ff_features, sublayer_param, fixed_param
    )
    for fcf, ff, sub, fix in zip(fcf_list, ff_list, sub_list, fix_list):
        command_fit_cmr(
            study,
            fit,
            fcf,
            ff,
            sublayers,
            sub,
            fix,
            **kwargs,
        )


def command_xval_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    sublayers,
    subpar,
    fixed,
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
        fcf_features, ff_features, sublayers, subpar, fixed
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

    if subpar:
        opts += f' -p {subpar}'
    if fixed:
        opts += f' -f {fixed}'
    full_dir = study_dir / study / 'fits' / fit / res_name

    print(f'cfr-xval-cmr {inputs} {fcf_features} {ff_features} {full_dir} {opts}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("fcf_features")
@click.argument("ff_features")
@click.option("--sublayers/--no-sublayers", default=False)
@click.option(
    "--sublayer-param",
    "-p",
    help="parameters free to vary between sublayers (e.g., B_enc-B_rec)",
)
@click.option(
    "--fixed-param",
    "-f",
    help="dash-separated list of values for fixed parameters (e.g., B_enc_cat=1)",
)
@click.option(
    "--n-folds",
    "-d",
    type=int,
    help="number of cross-validation folds to run",
)
@click.option(
    "--fold-key",
    "-k",
    help="events column to use when defining cross-validation folds",
)
@click.option(
    "--n-reps",
    "-n",
    type=int,
    default=1,
    help="number of times to replicate the search",
)
@click.option(
    "--n-jobs", "-j", type=int, default=1, help="number of parallel jobs to use"
)
@click.option("--tol", "-t", type=float, default=0.00001, help="search tolerance")
def plan_xval_cmr(
    study,
    fit,
    fcf_features,
    ff_features,
    sublayers,
    sublayer_param,
    fixed_param,
    **kwargs,
):
    """Print command lines for fitting multiple models."""
    fcf_list, ff_list, sub_list, fix_list = expand_variants(
        fcf_features, ff_features, sublayer_param, fixed_param
    )
    for fcf, ff, sub, fix in zip(fcf_list, ff_list, sub_list, fix_list):
        command_xval_cmr(
            study,
            fit,
            fcf,
            ff,
            sublayers,
            sub,
            fix,
            **kwargs,
        )


def command_sim_cmr(study, fit, model, n_rep=1):
    """Generate command line arguments for simulating CMR."""
    study_dir, data_file, patterns_file = framework.get_study_paths(study)
    fit_dir = study_dir / study / 'fits' / fit / model
    if not fit_dir.exists():
        raise IOError(f'Fit directory does not exist: {fit_dir}')
    print(f'cfr-sim-cmr {data_file} {patterns_file} {fit_dir} -r {n_rep}')


@click.command()
@click.argument("study")
@click.argument("fit")
@click.argument("models")
@click.option(
    "--n-sim-reps",
    "-r",
    type=int,
    default=1,
    help="number of experiment replications to simulate",
)
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
    print(f'cfr-plot-fit -e {ext} {data_file} {patterns_file} {fit_dir}')


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
@click.argument("out_dir", type=click.Path())
@click.argument("split_dirs", nargs=-1, type=click.Path(exists=True))
def join_xval(out_dir, split_dirs):
    """Join a split-up cross-validation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    param_file = out_dir / "parameters.json"
    search_file = out_dir / "xval_search.csv"
    xval_file = out_dir / "xval.csv"
    log_file = out_dir / "log_xval.txt"
    log = ""
    search_list = []
    xval_list = []
    for split_dir in split_dirs:
        print(f"Loading data from {split_dir}.")
        split_dir = Path(split_dir)

        # copy parameters file
        if not param_file.exists():
            split_param_file = split_dir / "parameters.json"
            if split_param_file.exists():
                shutil.copy(split_param_file, param_file)
            else:
                raise IOError(f"Parameters file does not exist: {split_param_file}")

        # add to log
        split_log_file = split_dir / "log_xval.txt"
        if not split_log_file.exists():
            raise IOError(f"Log file does not exist: {split_log_file}")
        log += split_log_file.read_text()

        # add to search
        split_search_file = split_dir / "xval_search.csv"
        if not split_search_file.exists():
            raise IOError(f"Search file does not exist: {split_search_file}")
        split_search = pd.read_csv(split_search_file)
        search_list.append(split_search)

        # add to xval
        split_xval_file = split_dir / "xval.csv"
        if not split_xval_file.exists():
            raise IOError(f"X-val file does not exist: {split_xval_file}")
        split_xval = pd.read_csv(split_xval_file)
        xval_list.append(split_xval)

    # write out concatenated files
    print(f"Writing data to {out_dir}.")
    log_file.write_text(log)
    search = pd.concat(search_list, ignore_index=True)
    search.sort_values(["fold", "subject", "rep"]).to_csv(search_file, index=False)
    xval = pd.concat(xval_list, ignore_index=True)
    xval.sort_values(["fold", "subject"]).to_csv(xval_file, index=False)
