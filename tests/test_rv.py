"""Tests for random variables."""


import jax.numpy as jnp
import pytest

import tornado

# Common fixtures


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


# Tests for multivariate normal


@pytest.fixture
def multivariate_normal(mean, cov_sqrtm):
    return tornado.rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)


def test_multivariate_normal_type(multivariate_normal):
    assert isinstance(multivariate_normal, tornado.rv.MultivariateNormal)


def test_multivariate_normal_cov(multivariate_normal):
    SC = multivariate_normal.cov_sqrtm
    C = multivariate_normal.cov
    assert jnp.allclose(C, SC @ SC.T)


# Tests for batched multivariate normal


@pytest.fixture
def batched_mean(mean, batch_size):
    return jnp.stack([mean] * batch_size).T


@pytest.fixture
def batched_cov_sqrtm(cov_sqrtm, batch_size):
    return jnp.stack([cov_sqrtm] * batch_size)


@pytest.fixture
def batched_multivariate_normal(batched_mean, batched_cov_sqrtm):
    return tornado.rv.BatchedMultivariateNormal(
        mean=batched_mean, cov_sqrtm=batched_cov_sqrtm
    )


def test_batched_multivariate_normal_type(batched_multivariate_normal):
    assert isinstance(batched_multivariate_normal, tornado.rv.BatchedMultivariateNormal)


def test_batched_multivariate_shapes_mean(
    batched_multivariate_normal, batch_size, dimension
):
    mean = batched_multivariate_normal.mean
    assert mean.shape == (dimension, batch_size)


def test_batched_multivariate_shapes_cov_sqrtm(
    batched_multivariate_normal, batch_size, dimension
):
    cov_sqrtm = batched_multivariate_normal.cov_sqrtm
    assert cov_sqrtm.shape == (batch_size, dimension, dimension)


def test_batched_multivariate_shapes_cov(
    batched_multivariate_normal, batch_size, dimension
):
    cov = batched_multivariate_normal.cov
    assert cov.shape == (batch_size, dimension, dimension)


def test_batched_multivariate_normal_cov(batched_multivariate_normal):
    batched_SC = batched_multivariate_normal.cov_sqrtm
    batched_SC_T = jnp.transpose(batched_SC, axes=(0, 2, 1))
    batched_C = batched_multivariate_normal.cov
    assert jnp.allclose(batched_C, batched_SC @ batched_SC_T)


# Tests for matrix normal


@pytest.fixture
def matrix_normal(mean, cov_sqrtm):
    return tornado.rv.MatrixNormal(
        mean=mean, cov_sqrtm_1=cov_sqrtm, cov_sqrtm_2=cov_sqrtm
    )


def test_matrix_normal_type(matrix_normal):
    assert isinstance(matrix_normal, tornado.rv.MatrixNormal)


def test_matrix_normal_cov_1(matrix_normal):
    SC = matrix_normal.cov_sqrtm_1
    C = matrix_normal.cov_1
    assert jnp.allclose(C, SC @ SC.T)


def test_matrix_normal_cov_2(matrix_normal):
    SC = matrix_normal.cov_sqrtm_2
    C = matrix_normal.cov_2
    assert jnp.allclose(C, SC @ SC.T)
