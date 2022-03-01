"""Test implementation of logistic ridge regression classifier."""

import numpy as np
from cfr import decode
import pytest


@pytest.fixture()
def patterns():
    vectors = np.array(
        [
            [-0.1022, -1.2141, 1.5442],
            [-0.2414, -1.1135, 0.0859],
            [0.3192, -0.0068, -1.4916],
            [0.3129, 1.5326, -0.7423],
            [-0.8649, -0.7697, -1.0616],
            [-0.0301, 0.3714, 2.3505],
            [-0.1649, -0.2256, -0.6156],
            [0.6277, 1.1174, 0.7481],
            [1.0933, -1.0891, -0.1924],
            [1.1093, 0.0326, 0.8886],
            [-0.8637, 0.5525, -0.7648],
            [0.0774, 1.1006, -1.4023],
        ]
    )
    labels = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2, 3, 3])
    chunks = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2])
    data = {'vectors': vectors, 'labels': labels, 'chunks': chunks}
    return data


def test_logreg1(patterns):
    train = patterns['chunks'] == 1
    labels = patterns['labels'][train]
    x = patterns['vectors'][train, :].T
    expected = np.array([[0.0016], [-0.1482], [0.0955]])
    y = np.zeros((1, len(labels)))
    y[:, labels == 1] = 1
    l = 10
    tol = 0.0001
    max_rounds = 5000
    w, ll, n = decode.logistic_regression(x, y, l, tol, max_rounds)
    np.testing.assert_allclose(w, expected, atol=0.0001)
