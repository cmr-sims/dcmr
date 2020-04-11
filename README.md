# cmr_cfr
Analyze categorized free recall data using CMR.

## Installation

To install Python modules and scripts:

```bash
git clone https://github.com/vucml/cmr_cfr.git
cd cmr_cfr
python setup.py install
```

or use `python setup.py develop` if you want to make
edits to the source code and have them take effect
immediately.

## Making plots

To make plots, you will first need to export data to a
CSV file. See `frdata2table.m` in the Psifr project.

To run serial position and lag-CRP analyses, including
analyses that are conditional on category:

```bash
cfr_plot_data.py [path_to_csv] [output_directory]
```
