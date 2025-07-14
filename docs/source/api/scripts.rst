Scripts
=======

The script interface makes it possible to evaluate many model variants by specifying command-line options.
For a more flexible Python interface, see :py:mod:`dcmr.framework` and :py:mod:`dcmr.reports`.

To work with a dataset, you must first prepare a Psifr-formatted data table in CSV format
and an HDF5 file with patterns that can be used in the model's connection weights.

Scripts allow you to estimate parameters for individual subjects, make an HTML report with
fit diagnostics, use cross-validation to estimate model goodness of fit, and run a simulation
based on an existing fit but with adjusted parameters.

.. toctree::
    :maxdepth: 2

    /api/scripts/fit_cmr
    /api/scripts/plot_fit
    /api/scripts/xval_cmr
    /api/scripts/adjust_sim
