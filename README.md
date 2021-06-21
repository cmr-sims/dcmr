# cmr_cfr
Analyze categorized free recall data using CMR.

## Installation

It's recommended that you first set up a conda environment or Python virtual environment. For example, using Conda:

```bash
conda create -n cfr python=3.8
conda activate cfr
```

To install the latest code from GitHub:
```bash
pip install git+git://github.com/vucml/cmr_cfr
```

To install the code in editable mode for development
(changes to the local copy of the code will be applied to the installed package without having to reinstall):

```bash
git clone https://github.com/vucml/cmr_cfr.git
pip install -e cmr_cfr
```

## Making plots

To make plots, you will first need to export data to a
CSV file. See `frdata2table.m` in the Psifr project.

To run serial position and lag-CRP analyses, including
analyses that are conditional on category:

```bash
cfr_plot_data.py [path_to_csv] [output_directory]
```
