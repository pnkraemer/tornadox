"""Tests for Gaussian filtering and smoothing routines."""


import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def n():
    return 4


@pytest.fixture
def m(n):
    return jnp.arange(1, 1 + n)


@pytest.fixture
def sc(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def phi(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def sq(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def h(n):
    return jnp.eye(n // 2, n)


@pytest.fixture
def b(n):
    return jnp.arange(1, 1 + n // 2)


@pytest.fixture
def data(n):
    return jnp.arange(10, 10 + n // 2)


def test_filter_step(m, sc, phi, sq, h, b, data):

    m, sc, sgain, mp, scp = tornado.kalman.filter_step(
        m=m,
        sc=sc,
        phi=phi,
        sq=sq,
        h=h,
        b=b,
        data=data,
    )
    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)
    assert isinstance(sgain, jnp.ndarray)
    assert isinstance(mp, jnp.ndarray)
    assert isinstance(scp, jnp.ndarray)
