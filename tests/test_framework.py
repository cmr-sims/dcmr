"""Test code implementing the model framework."""

from cfr import framework


def test_weight_param1():
    """Weight parameters with one sublayer."""
    wp = framework.model_variant(['loc'], None, sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['fc'] == {(('task', 'item'), ('task', 'loc')): 'Dfc * loc'}
    assert wp.weights['cf'] == {(('task', 'item'), ('task', 'loc')): 'Dcf * loc'}


def test_weight_param2():
    """Weight parameters with two sublayers."""
    wp = framework.model_variant(['loc', 'cat'], None, sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dfc * w_cat * cat',
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('task', 'loc')): 'Dcf * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dcf * w_cat * cat',
    }


def test_weight_param3():
    """Weight parameters with three sublayers."""
    wp = framework.model_variant(['loc', 'cat', 'use'], None, sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('task', 'use')): 'Dfc * w_use * use',
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('task', 'loc')): 'Dcf * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dcf * w_cat * cat',
        (('task', 'item'), ('task', 'use')): 'Dcf * w_use * use',
    }


def test_item_weight_param1():
    """Item-item weight parameters with one sublayer."""
    wp = framework.model_variant(['loc'], ['loc'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {('task', 'item'): 'Dff * (loc)'}


def test_item_weight_param2():
    """Item-item weight parameters with two sublayers."""
    wp = framework.model_variant(['loc'], ['loc', 'cat'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat)'}


def test_item_weight_param3():
    """Item-item weight parameters with three sublayers."""
    wp = framework.model_variant(['loc'], ['loc', 'cat', 'use'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {
        ('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat + s_use * use)'
    }


def test_weight_param_sublayers1():
    """Using one sublayer does not set sublayer-dependent learning."""
    wp = framework.model_variant(['loc'], None, sublayers=True)
    assert 'Lfc_loc' not in wp.dependent
    assert 'Lcf_loc' not in wp.dependent
    assert wp.sublayers == {'f': ['task'], 'c': ['loc']}
    assert wp.sublayer_param == {}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'Dfc * loc',
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('loc', 'item')): 'Dcf * loc',
    }


def test_weight_param_sublayers3():
    """Using three sublayers sets sublayer-dependent learning."""
    wp = framework.model_variant(['loc', 'cat', 'use'], None, sublayers=True)
    assert wp.sublayers == {'f': ['task'], 'c': ['loc', 'cat', 'use']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dfc * w_use * use',
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('loc', 'item')): 'Dcf * w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dcf * w_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dcf * w_use * use',
    }
    assert wp.dependent['Lfc_loc'] == 'Lfc * w_loc'
    assert wp.dependent['Lcf_loc'] == 'Lcf * w_loc'
    assert wp.dependent['Lfc_cat'] == 'Lfc * w_cat'
    assert wp.dependent['Lcf_cat'] == 'Lcf * w_cat'
    assert wp.dependent['Lfc_use'] == 'Lfc * w_use'
    assert wp.dependent['Lcf_use'] == 'Lcf * w_use'
    assert wp.sublayer_param == {
        'c': {
            'loc': {'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc'},
            'cat': {'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat'},
            'use': {'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use'},
        }
    }


def test_param_sublayers1():
    """Free sublayer parameter with one sublayer."""
    wp = framework.model_variant(
        ['loc'], None, sublayers=True, sublayer_param=['B_enc']
    )
    assert 'B_enc_loc' in wp.free
    assert 'B_enc' not in wp.free
    assert wp.sublayers == {'f': ['task'], 'c': ['loc']}
    assert wp.sublayer_param == {'c': {'loc': {'B_enc': 'B_enc_loc'}}}


def test_param_sublayers3():
    """Free sublayer parameter with three sublayers."""
    wp = framework.model_variant(
        ['loc', 'cat', 'use'], None, sublayers=True, sublayer_param=['B_enc']
    )
    assert 'B_enc_loc' in wp.free
    assert 'B_enc' not in wp.free
    assert wp.sublayers == {'f': ['task'], 'c': ['loc', 'cat', 'use']}
    assert wp.sublayer_param == {
        'c': {
            'loc': {'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc', 'B_enc': 'B_enc_loc'},
            'cat': {'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat', 'B_enc': 'B_enc_cat'},
            'use': {'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use', 'B_enc': 'B_enc_use'},
        }
    }


def test_learning_param_sublayers():
    """Free sublayer learning parameter modifies weights."""
    wp = framework.model_variant(
        ['loc', 'cat', 'use'], None, sublayers=True, sublayer_param=['Lcf']
    )
    assert 'Lcf_loc_raw' in wp.free
    assert 'Lcf' not in wp.free
    assert wp.dependent['Lfc_loc'] == 'Lfc * w_loc'
    assert wp.dependent['Lcf_loc'] == 'Lcf_loc_raw * w_loc'
    assert wp.dependent['Dfc'] == '1 - Lfc'
    assert wp.dependent['Dcf_loc'] == '1 - Lcf_loc_raw'
    assert wp.weights['fc'][(('task', 'item'), ('loc', 'item'))] == 'Dfc * w_loc * loc'
    assert (
        wp.weights['cf'][(('task', 'item'), ('loc', 'item'))] == 'Dcf_loc * w_loc * loc'
    )
    assert wp.sublayer_param == {
        'c': {
            'loc': {'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc'},
            'cat': {'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat'},
            'use': {'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use'},
        }
    }
