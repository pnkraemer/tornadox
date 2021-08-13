"""Tests for random variables."""


import tornado
import jax.numpy



def test_rv():
    """Random variables work as expected."""
    mean = jax.numpy.array([1., 2.])
    cov_cholesky = jax.numpy.array([[1., 0.], [1., 1.]])
    normal = tornado.rv.MultivariateNormal(mean=mean, cov_cholesky=cov_cholesky)

    assert isinstance(normal, tornado.rv.MultivariateNormal)
    assert jax.numpy.allclose(normal.cov, cov_cholesky @ cov_cholesky.T)
