Framework
=========

.. currentmodule:: dcmr.framework

Model configuration
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: api/

    model_variant

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
