# Set up Bash environment for running project scripts.

if [[ $USER = morton ]]; then
    . .venv/bin/activate
    export STUDYDIR=$HOME/Dropbox/work/cmr_cfr
    export CFR_RESULTS=$HOME/Dropbox/work/cmr_cfr/cfr
    export CFR_FITS=$HOME/Dropbox/work/cmr_cfr/cfr/fits/v5
    export CFR_FIGURES=$HOME/Dropbox/work/cmr_cfr/cfr/figs/v2
else
    . /work/03206/mortonne/lonestar/venv/cfr/bin/activate
    unset PYTHONPATH
    export STUDYDIR=/work/03206/mortonne/lonestar/cmr_cfr
fi

export BATCHDIR=$STUDYDIR/batch
export SUBJNOS=1:2:3:5:8:11:16:18:22:23:24:25:27:28:29:31:32:33:34:35:37:38:40:41:42:43:44:45:46
