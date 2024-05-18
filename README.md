# DCMR

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

See the 
[Analysis protocol](https://github.com/vucml/cmr_cfr/wiki/CFR-Analysis-Protocol)
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
[notebook](https://github.com/cmr-sims/dcmr/blob/master/jupyter/create_patterns.ipynb).
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
and simulated data generated using the model with the best-fitting parameters.
The simulated data are saved in a Psifr-format CSV file and can be analyzed just like real observed data.
Run `dcmr-fit -h` to see the many options for configuring model variants.

### Evaluating a fit

After running a fit, you'll want to evaluate how well the model captures the observed data.
To create an HTML report comparing observed data to simulated data from a fitted model,
using analyses like the serial position curve, probability of first recall, and conditional response probability by lag: 

```bash
dcmr-plot-fit data.csv patterns.hdf5 cmr_fit
```

After the script runs, you should have a `report.html` file in the fit directory that you can open using a web browser.

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
