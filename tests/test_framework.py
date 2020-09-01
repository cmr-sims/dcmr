"""Test code implementing the model framework."""

from cfr import framework


def test_scaling_param_w1():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc'])
    assert spar == {'loc': None}
    assert wp.free == {}
    assert wp.dependent == {}


def test_scaling_param_w2():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat'])
    assert spar == {'loc': 'w_loc', 'cat': 'w_cat'}
    assert wp.free == {'w0': (0, 1)}
    assert wp.dependent == {
        'wr_loc': 'w0',
        'wr_cat': '1 - w0',
        'w_loc': 'wr_loc / sqrt(wr_loc**2 + wr_cat**2)',
        'w_cat': 'wr_cat / sqrt(wr_loc**2 + wr_cat**2)',
    }


def test_scaling_param_w3():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat', 'use'], upper=100)
    assert spar == {'loc': 'w_loc', 'cat': 'w_cat', 'use': 'w_use'}
    assert wp.free == {'w0': (0, 1), 'w1': (0, 100)}
    assert wp.dependent == {
        'wr_loc': 'w0',
        'wr_cat': '1 - w0',
        'wr_use': 'w1',
        'w_loc': 'wr_loc / sqrt(wr_loc**2 + wr_cat**2 + wr_use**2)',
        'w_cat': 'wr_cat / sqrt(wr_loc**2 + wr_cat**2 + wr_use**2)',
        'w_use': 'wr_use / sqrt(wr_loc**2 + wr_cat**2 + wr_use**2)',
    }


def test_scaling_param_s1():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc'])
    assert spar == {'loc': None}
    assert wp.free == {}
    assert wp.dependent == {}


def test_scaling_param_s2():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc', 'cat'])
    assert spar == {'loc': 's_loc', 'cat': 's_cat'}
    assert wp.free == {'s0': (0, 1)}
    assert wp.dependent == {
        'sr_loc': 's0',
        'sr_cat': '1 - s0',
        's_loc': 'sr_loc / (sr_loc + sr_cat)',
        's_cat': 'sr_cat / (sr_loc + sr_cat)',
    }


def test_scaling_param_s3():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc', 'cat', 'use'], upper=100)
    assert spar == {'loc': 's_loc', 'cat': 's_cat', 'use': 's_use'}
    assert wp.free == {'s0': (0, 1), 's1': (0, 100)}
    assert wp.dependent == {
        'sr_loc': 's0',
        'sr_cat': '1 - s0',
        'sr_use': 's1',
        's_loc': 'sr_loc / (sr_loc + sr_cat + sr_use)',
        's_cat': 'sr_cat / (sr_loc + sr_cat + sr_use)',
        's_use': 'sr_use / (sr_loc + sr_cat + sr_use)',
    }


def test_region_weights():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat', 'use'])
    wp.set_region_weights('fc', spar, 'Dfc')
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('task', 'use')): 'Dfc * w_use * use',
    }


def test_sublayer_weights():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat', 'use'])
    wp.set_sublayer_weights('fc', spar, 'Dfc')
    assert wp.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dfc * w_use * use',
    }


def test_item_weights():
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc', 'cat', 'use'])
    wp.set_item_weights(spar, 'Dff')
    assert wp.weights['ff'] == {
        ('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat + s_use * use)',
    }


def test_weight_param1():
    wp = framework.model_variant(['loc'], None, sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * loc'
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('task', 'loc')): 'Dcf * loc'
    }


def test_weight_param2():
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
    wp = framework.model_variant(['loc'], ['loc'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {('task', 'item'): 'Dff * (loc)'}


def test_item_weight_param2():
    wp = framework.model_variant(['loc'], ['loc', 'cat'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {
        ('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat)'
    }


def test_item_weight_param3():
    wp = framework.model_variant(['loc'], ['loc', 'cat', 'use'], sublayers=False)
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['ff'] == {
        ('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat + s_use * use)'
    }


def test_weight_param_sublayers():
    wp = framework.model_variant(['loc', 'cat', 'use'], None, sublayers=True,
                                 sublayer_param=['B_enc'])
    assert 'B_enc_loc' in wp.free
    assert 'B_enc' not in wp.free
    assert wp.sublayers == {'f': ['task'], 'c': ['loc', 'cat', 'use']}
    assert wp.sublayer_param == {'c': {
        'loc': {'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc', 'B_enc': 'B_enc_loc'},
        'cat': {'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat', 'B_enc': 'B_enc_cat'},
        'use': {'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use', 'B_enc': 'B_enc_use'},
    }}


def test_learning_param_sublayers():
    wp = framework.model_variant(['loc', 'cat', 'use'], None, sublayers=True,
                                 sublayer_param=['Lcf'])
    assert 'Lcf_loc_raw' in wp.free
    assert 'Lcf' not in wp.free
    assert wp.dependent['Lfc_loc'] == 'Lfc * w_loc'
    assert wp.dependent['Lcf_loc'] == 'Lcf_loc_raw * w_loc'
    assert wp.dependent['Dfc'] == '1 - Lfc'
    assert wp.dependent['Dcf_loc'] == '1 - Lcf_loc_raw'
    assert wp.weights['fc'][(('task', 'item'), ('loc', 'item'))] == 'Dfc * w_loc * loc'
    assert wp.weights['cf'][(('task', 'item'), ('loc', 'item'))] == 'Dcf_loc * w_loc * loc'
    assert wp.sublayer_param == {'c': {
        'loc': {'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc'},
        'cat': {'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat'},
        'use': {'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use'}
    }}
