# DCMR

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.11214414.svg)](https://doi.org/10.5281/zenodo.11214414)

Analyze free recall data using the distributed context maintenance and retrieval (DCMR) model.

## Installation

It's recommended that you first set up a conda environment or Python virtual environment. For example, using Conda:

```bash
conda create -n dcmr python=3.10
conda activate dcmr
```

To install the latest code from GitHub:
```bash
pip install git+git://github.com/cmr-sims/dcmr
```

To install the code in editable mode for development
(changes to the local copy of the code will be applied to the installed package without having to reinstall):

```bash
git clone https://github.com/cmr-sims/dcmr.git
pip install -e dcmr
```

## Usage

The DCMR package is designed for flexible implementation and evaluation of many model variants,
which may be applied to any free-recall paradigm with data stored in Psifr format.
See the [documentation](https://dcmr.readthedocs.io/en/latest/) for details about
running parameter searches, comparing models, and evaluating goodness of fit.

See the 
[Analysis protocol](https://github.com/cmr-sims/dcmr/wiki)
for details of the analyses for the DCMR paper (in preparation). 

### Data

Fitting data or running simulations requires having data in 
[Psifr format](https://psifr.readthedocs.io/en/stable/guide/import.html)
in a CSV file. Data in EMBAM/behavioral toolbox MAT-file format can be converted using
[`frdata2table`](https://github.com/mortonne/psifr/blob/master/matlab/frdata2table.m).

### Patterns

The CMR model uses weight matrices or "patterns" to define the strength of connections between layers of the model network. 
To run simulations, you must first define these patterns and save them to a patterns file. 
See the pattern creation 
[notebook](https://github.com/cmr-sims/dcmr/blob/main/jupyter/create_patterns.ipynb).
A model can make use of multiple patterns representing different types of pre-existing knowledge about a set of stimuli, 
such as their category or detailed semantic features. 

### Fitting data

To fit a variant of the CMR model to a dataset, use `dcmr-fit`. 
For example:

```bash
dcmr-fit data.csv patterns.hdf5 loc none cmr_fit
```

will fit a model with localist weights (as defined in the patterns file) to a dataset and save out the fit results to a `cmr_fit` directory. 
Results include the best-fitting parameters for each subject, 
the log likelihood of the observed data according to the model with those parameters,
simulated data generated using the model with the best-fitting parameters,
and an HTML report with fit diagnostics.
The simulated data are saved in a Psifr-format CSV file and can be analyzed just like real observed data
using [Psifr](https://dcmr.readthedocs.io/en/latest/).
Run `dcmr-fit --help` to see the many options for configuring model variants.

### Evaluating a fit

After running a fit, you'll want to evaluate how well the model captures the observed data.
The `dcmr-fit` script will automatically create a fit diagnostics report comparing observed data to simulated data from a fitted model,
using analyses like the serial position curve, probability of first recall, and conditional response probability by lag.

The `dcmr-plot-fit` script allows you to create additional reports after a fit has finished.
For example, to recreate the report for the fit in the `cmr_fit` directory:

```bash
dcmr-plot-fit data.csv patterns.hdf5 cmr_fit
```

After the script runs, you should have a `report.html` file in the fit directory that you can open using a web browser.

Run `dcmr-plot-fit --help` to see options. To separately examine different conditions, set the `--data-filter` option
to specify an expression that will yield the correct trials; for example `--data-filter 'condition == 1'` will plot
just condition 1 trials. Use `--report-name` to indicate a subdirectory that the results should be placed int.

### Using cross-validation to evaluate a model

Cross-validation, where a model is fit to a subset of data and tested on a left-out set,
can be used to determine how well a model captures reliable patterns in the data.

Cross-validation involves dividing data into "folds". 
This can be done by specifying either a number of random folds to use (lists are randomly divided into folds)
or a column within the data (e.g., the session number) that can be used to group lists into folds. For example:

```bash
dcmr-xval data.csv patterns.hdf5 loc none cmr_fit -k session
```

Run `dcmr-xval -h` to see all options.
