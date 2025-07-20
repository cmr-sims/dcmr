#!/bin/bash
#
# Run tests of DCMR command line interface tools.

cfr_data_dir=$1
out_dir=$2

data=${cfr_data_dir}/cfr_data.csv
patterns=${cfr_data_dir}/cfr_patterns.hdf5

opt="--sublayers -n 1 -j 1 -i 1-2-3 -t 0.01"
fit_opt="$opt -r 50"
xval_opt="$opt -k session"

# test models with various sublayers and weights
dcmr-fit "$data" "$patterns" loc none "${out_dir}/fcf-loc" $fit_opt
dcmr-fit "$data" "$patterns" cat none "${out_dir}/fcf-cat" $fit_opt
dcmr-fit "$data" "$patterns" use none "${out_dir}/fcf-use" $fit_opt
dcmr-fit "$data" "$patterns" loc-cat-use none "${out_dir}/fcf-loc-cat-use" $fit_opt
dcmr-fit "$data" "$patterns" loc cat-use "${out_dir}/fcf-loc_ff-cat-use" $fit_opt

# test report options
dcmr-plot-fit "$data" "$patterns" "${out_dir}/fcf-loc" -r report_manual
dcmr-plot-fit "$data" "$patterns" "${out_dir}/fcf-loc" -r report_nocat --no-category
dcmr-plot-fit "$data" "$patterns" "${out_dir}/fcf-loc" -r report_nosim --no-similarity
dcmr-plot-fit "$data" "$patterns" "${out_dir}/fcf-loc" -r report_nocat_nosim --no-category --no-similarity
dcmr-fit "$data" "$patterns" loc none "${out_dir}/fcf-loc_rep-nocat" $fit_opt --no-category

# cross-validation
dcmr-xval "$data" "$patterns" loc none "${out_dir}/fcf-loc" $xval_opt
