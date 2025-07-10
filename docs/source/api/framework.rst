Framework
=========

.. currentmodule:: dcmr.framework

The :code:`framework` module specifies the operation of the DCMR model using the CyMR engine. 
Model variants may be configured using :py:func:`model_variant`. Model specifications
are defined by :py:class:`WeightParameters` objects, which may be stored in JSON files.

This module also provides functions for running parameter searches, evaluating models using
cross-validation, and working with fit and cross-validation results.

Model configuration
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    model_variant

Evaluating models
~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    run_fit
    run_xval

Fit results
~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    read_fit_param
    read_fit_weights
    read_model_spec
    read_model_specs
    read_model_fits
    read_model_xvals
    read_model_sims
    get_sim_models
    compare_fit

Model parameters
~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    WeightParameters
    WeightParameters.set_scaling_param
    WeightParameters.set_intercept_param
    WeightParameters.set_region_weights
    WeightParameters.set_sublayer_weights
    WeightParameters.set_item_weights
    WeightParameters.set_learning_sublayer_param
    WeightParameters.set_weight_sublayer_param
    WeightParameters.set_free_sublayer_param
