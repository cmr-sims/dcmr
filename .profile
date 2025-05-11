# Set up Bash environment for running project scripts.

if [[ $USER = morton || $USER = nmorton ]]; then
    . .venv/bin/activate
    export STUDYDIR=$HOME/Dropbox/work/dcmr
    export CFR_RESULTS=$HOME/Dropbox/work/dcmr/cfr
    export CFR_FITS=$HOME/Dropbox/work/dcmr/cfr/fits/v5
    export CFR_FIGURES=$HOME/Dropbox/work/dcmr/cfr/figs/v3
    export PEERS_RESULTS=$HOME/Dropbox/work/dcmr/peers
    export PEERS_FITS=$HOME/Dropbox/work/dcmr/peers/fits/v5
    export PEERS_FIGURES=$HOME/Dropbox/work/dcmr/peers/figs/v1
elif [[ $USER = mortonne ]]; then
    . /work/03206/mortonne/lonestar/venv/cfr/bin/activate
    unset PYTHONPATH
    export STUDYDIR=/work/03206/mortonne/lonestar/dcmr
else
    echo "Warning: cannot set environment for unknown user $USER. Edit .profile to support your environment."
fi

export BATCHDIR=$STUDYDIR/batch
export SUBJNOS=1:2:3:5:8:11:16:18:22:23:24:25:27:28:29:31:32:33:34:35:37:38:40:41:42:43:44:45:46
