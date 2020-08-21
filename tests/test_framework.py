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
