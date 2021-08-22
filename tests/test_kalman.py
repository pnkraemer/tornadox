"""Tests for Gaussian filtering and smoothing routines."""


import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def n():
    return 4


@pytest.fixture
def m_1d(n):
    return jnp.arange(1, 1 + n)


@pytest.fixture
def sc_1d(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def phi_1d(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def sq_1d(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def h_1d(n):
    return jnp.eye(n // 2, n)


@pytest.fixture
def b_1d(n):
    return jnp.arange(1, 1 + n // 2)


@pytest.fixture
def data(n):
    return jnp.arange(10, 10 + n // 2)


def test_filter_step(m_1d, sc_1d, phi_1d, sq_1d, h_1d, b_1d, data):

    m, sc, sgain = tornado.kalman.filter_step_1d(
        m_1d=m_1d,
        sc_1d=sc_1d,
        phi_1d=phi_1d,
        sq_1d=sq_1d,
        h_1d=h_1d,
        b_1d=b_1d,
        data=data,
    )
    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)
    assert isinstance(sgain, jnp.ndarray)
