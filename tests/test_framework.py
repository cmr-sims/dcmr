"""Test code implementing the model framework."""

from cfr import framework


def test_weight_param1():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc'], sublayers=False)
    assert wp.free == {}
    assert wp.dependent == {}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    weights = {
        (('task', 'item'), ('task', 'loc')): 'loc',
    }
    assert wp.weights['fc'] == weights
    assert wp.weights['cf'] == weights


def test_weight_param2():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc', 'cat'], sublayers=False)

    assert wp.free == {'w0': (0, 1)}
    assert wp.dependent == {'w_loc': 'w0', 'w_cat': '1 - w0'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    weights = {
        (('task', 'item'), ('task', 'loc')): 'w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'w_cat * cat',
    }
    assert wp.weights['fc'] == weights
    assert wp.weights['cf'] == weights


def test_weight_param3():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc', 'cat', 'use'], sublayers=False)

    assert wp.free == {'w0': (0, 1), 'w1': (0, 100)}
    assert wp.dependent == {'w_loc': 'w0', 'w_cat': '1 - w0', 'w_use': 'w1'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    weights = {
        (('task', 'item'), ('task', 'loc')): 'w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'w_cat * cat',
        (('task', 'item'), ('task', 'use')): 'w_use * use',
    }
    assert wp.weights['fc'] == weights
    assert wp.weights['cf'] == weights


def test_item_weight_param1():
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc'], sublayers=False)

    assert wp.free == {}
    assert wp.dependent == {}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {'ff': {('task', 'item'): 'loc'}}


def test_item_weight_param2():
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc', 'cat'], sublayers=False)

    assert wp.free == {'s0': (0, 1)}
    assert wp.dependent == {'s_loc': 's0', 's_cat': '1 - s0'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {
        'ff': {('task', 'item'): 's_loc * loc + s_cat * cat'}
    }


def test_item_weight_param3():
    wp = framework.WeightParameters()
    wp.set_weight_param('ff', ['loc', 'cat', 'use'], sublayers=False)

    assert wp.free == {'s0': (0, 1), 's1': (0, 100)}
    assert wp.dependent == {'s_loc': 's0', 's_cat': '1 - s0', 's_use': 's1'}
    assert wp.sublayers == {'f': ['task'], 'c': ['task']}
    assert wp.weights == {
        'ff': {('task', 'item'): 's_loc * loc + s_cat * cat + s_use * use'}
    }


def test_weight_param_sublayers():
    wp = framework.WeightParameters()
    wp.set_weight_param('fcf', ['loc', 'cat', 'use'], sublayers=True)

    assert wp.free == {
        'B_enc_loc': (0, 1),
        'B_rec_loc': (0, 1),
        'B_enc_cat': (0, 1),
        'B_rec_cat': (0, 1),
        'B_enc_use': (0, 1),
        'B_rec_use': (0, 1),
    }
    assert wp.sublayers == {'f': ['task'], 'c': ['loc', 'cat', 'use']}
    weights = {
        (('task', 'item'), ('loc', 'item')): 'loc',
        (('task', 'item'), ('cat', 'item')): 'cat',
        (('task', 'item'), ('use', 'item')): 'use',
    }
    assert wp.weights['fc'] == weights
    assert wp.weights['cf'] == weights
    assert wp.sublayer_param['c'] == {
        'loc': {'B_enc': 'B_enc_loc', 'B_rec': 'B_rec_loc'},
        'cat': {'B_enc': 'B_enc_cat', 'B_rec': 'B_rec_cat'},
        'use': {'B_enc': 'B_enc_use', 'B_rec': 'B_rec_use'},
    }
