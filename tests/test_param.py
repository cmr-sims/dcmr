"""Test code defining CMR parameters."""

from dcmr import framework


def test_scaling_param_w1():
    """Set one item-context scaling parameter."""
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc'])
    assert spar == {'loc': None}
    assert wp.free == {}
    assert wp.dependent == {}


def test_scaling_param_w2():
    """Set two item-context scaling parameters."""
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
    """Set three item-context scaling parameters."""
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
    """Set one item-item scaling parameter."""
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc'])
    assert spar == {'loc': None}
    assert wp.free == {}
    assert wp.dependent == {}


def test_scaling_param_s2():
    """Set two item-item scaling parameters."""
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
    """Set three item-item scaling parameters."""
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
    """Set region weights with component scaling."""
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat', 'use'])
    wp.set_region_weights('fc', spar, 'Dfc')
    assert wp.weights['fc'] == {
        (('task', 'item'), ('task', 'loc')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('task', 'cat')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('task', 'use')): 'Dfc * w_use * use',
    }


def test_sublayer_weights():
    """Set sublayer weights with component scaling."""
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('vector', ['loc', 'cat', 'use'])
    wp.set_sublayer_weights('fc', spar, 'Dfc')
    assert wp.weights['fc'] == {
        (('task', 'item'), ('loc', 'item')): 'Dfc * w_loc * loc',
        (('task', 'item'), ('cat', 'item')): 'Dfc * w_cat * cat',
        (('task', 'item'), ('use', 'item')): 'Dfc * w_use * use',
    }


def test_item_weights():
    """Set item-item weights with component scaling."""
    wp = framework.WeightParameters()
    spar = wp.set_scaling_param('similarity', ['loc', 'cat', 'use'])
    wp.set_item_weights(spar, 'Dff')
    assert wp.weights['ff'] == {
        ('task', 'item'): 'Dff * (s_loc * loc + s_cat * cat + s_use * use)',
    }
