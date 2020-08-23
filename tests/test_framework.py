"""Test code implementing the model framework."""

from cfr import framework


def test_weight_param1():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc'])
    assert wp.free == {}
    assert wp.dependent == {}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * loc'
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('task', 'loc')): 'Dcf * loc'
    }


def test_weight_param2():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc', 'cat'])

    assert wp.free == {'w0': (0, 1)}
    assert wp.dependent == {'w_loc': 'w0', 'w_cat': '1 - w0'}
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
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc', 'cat', 'use'])

    assert wp.free == {'w0': (0, 1), 'w1': (0, 100)}
    assert wp.dependent == {'w_loc': 'w0', 'w_cat': '1 - w0', 'w_use': 'w1'}
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
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc'])

    assert wp.free == {'Dff': (0, 10)}
    assert wp.dependent == {}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {'ff': {('task', 'item'): 'Dff * (loc)'}}


def test_item_weight_param2():
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc', 'cat'])

    assert wp.free == {'Dff': (0, 10), 's0': (0, 1)}
    assert wp.dependent == {'s_loc': 's0', 's_cat': '1 - s0'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {
        'ff': {('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat)'}
    }


def test_item_weight_param3():
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc', 'cat', 'use'])

    assert wp.free == {'Dff': (0, 10), 's0': (0, 1), 's1': (0, 100)}
    assert wp.dependent == {'s_loc': 's0', 's_cat': '1 - s0', 's_use': 's1'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {
        'ff': {('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat + s_use * use)'}
    }


def test_weight_param_sublayers():
    wp = framework.WeightParameters()
    wp.set_weight_param_sublayer(['loc', 'cat', 'use'])

    assert wp.free == {
        'B_enc_loc': (0, 1),
        'B_rec_loc': (0, 1),
        'B_enc_cat': (0, 1),
        'B_rec_cat': (0, 1),
        'B_enc_use': (0, 1),
        'B_rec_use': (0, 1),
        'Lfc_loc': (0, 1),
        'Lcf_loc': (0, 1),
        'Lfc_cat': (0, 1),
        'Lcf_cat': (0, 1),
        'Lfc_use': (0, 1),
        'Lcf_use': (0, 1),
    }
    assert wp.dependent == {
        'Dfc_loc': '1 - Lfc_loc',
        'Dcf_loc': '1 - Lcf_loc',
        'Dfc_cat': '1 - Lfc_cat',
        'Dcf_cat': '1 - Lcf_cat',
        'Dfc_use': '1 - Lfc_use',
        'Dcf_use': '1 - Lcf_use',
    }
    assert wp.sublayers == {'f': ['task'], 'c': ['loc', 'cat', 'use']}
    assert wp.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'Dfc_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dfc_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dfc_use * use',
    }
    assert wp.weights['cf'] == {
        (('task', 'item'), ('loc', 'item')): 'Dcf_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dcf_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dcf_use * use',
    }
    assert wp.sublayer_param['c'] == {
        'loc': {'B_enc': 'B_enc_loc', 'B_rec': 'B_rec_loc',
                'Lfc': 'Lfc_loc', 'Lcf': 'Lcf_loc'},
        'cat': {'B_enc': 'B_enc_cat', 'B_rec': 'B_rec_cat',
                'Lfc': 'Lfc_cat', 'Lcf': 'Lcf_cat'},
        'use': {'B_enc': 'B_enc_use', 'B_rec': 'B_rec_use',
                'Lfc': 'Lfc_use', 'Lcf': 'Lcf_use'},
    }
