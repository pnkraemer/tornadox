"""Tests for random variables."""


import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def dimension():
    return 3


@pytest.fixture
def batch_size():
    return 5


@pytest.fixture
def mean(dimension):
    return jnp.arange(dimension)


@pytest.fixture
def cov_sqrtm(dimension):
    return jnp.arange(dimension ** 2).reshape((dimension, dimension))


@pytest.fixture
def multivariate_normal(mean, cov_sqrtm):
    return tornado.rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)


def test_multivariate_normal_type(multivariate_normal):
    assert isinstance(multivariate_normal, tornado.rv.MultivariateNormal)


def test_multivariate_normal_cov(multivariate_normal):
    SC = multivariate_normal.cov_sqrtm
    C = multivariate_normal.cov
    assert jnp.allclose(C, SC @ SC.T)
