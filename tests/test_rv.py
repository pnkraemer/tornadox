"""Tests for random variables."""


import jax.numpy

import tornado


def test_rv():
    """Random variables work as expected."""
    mean = jax.numpy.array([1.0, 2.0])
    cov_cholesky = jax.numpy.array([[1.0, 0.0], [1.0, 1.0]])
    normal = tornado.rv.MultivariateNormal(mean=mean, cov_cholesky=cov_cholesky)

    assert isinstance(normal, tornado.rv.MultivariateNormal)
    assert jax.numpy.allclose(normal.cov, cov_cholesky @ cov_cholesky.T)
