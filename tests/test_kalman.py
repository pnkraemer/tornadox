"""Tests for Gaussian filtering and smoothing routines."""


import jax.numpy as jnp
import pytest

import tornadox


@pytest.fixture
def n():
    return 4


@pytest.fixture
def m(n):
    return jnp.arange(1, 1 + n)


@pytest.fixture
def sc(n):
    return jnp.eye(n)  # + 0.001* jnp.arange(1, 1 + n ** 2).reshape((n, n)).T


@pytest.fixture
def phi(n):
    return jnp.triu(jnp.arange(1, 1 + n**2).reshape((n, n)).T)


@pytest.fixture
def sq(n):
    return jnp.eye(n)


@pytest.fixture
def h(n):
    return jnp.eye(n // 2, n)


@pytest.fixture
def b(n):
    return jnp.arange(1, 1 + n // 2)


@pytest.fixture
def data(n):
    return jnp.arange(10, 10 + n // 2)


@pytest.fixture
def filter_stepped(m, sc, phi, sq, h, b, data):

    return tornadox.kalman.filter_step(
        m=m,
        sc=sc,
        phi=phi,
        sq=sq,
        h=h,
        b=b,
        data=data,
    )


def test_filter_step_types(filter_stepped):
    m, sc, sgain, mp, scp, x = filter_stepped

    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)
    assert isinstance(sgain, jnp.ndarray)
    assert isinstance(mp, jnp.ndarray)
    assert isinstance(scp, jnp.ndarray)
    assert isinstance(x, jnp.ndarray)


def test_filter_step_shapes(filter_stepped, n):
    m, sc, sgain, mp, scp, x = filter_stepped

    assert m.shape == (n,)
    assert sc.shape == (n, n)
    assert sgain.shape == (n, n)
    assert mp.shape == (n,)
    assert scp.shape == (n, n)
    assert x.shape == (n, n)


@pytest.fixture
def smoother_stepped_traditional(m, sc, filter_stepped):
    m_fut, sc_fut, sgain, mp, scp, _ = filter_stepped

    m, sc = tornadox.kalman.smoother_step_traditional(
        m=m,
        sc=sc,
        m_fut=m_fut,
        sc_fut=sc_fut,
        sgain=sgain,
        mp=mp,
        scp=scp,
    )
    return m, sc


@pytest.fixture
def smoother_stepped_sqrt(m, sc, sq, filter_stepped):
    m_fut, sc_fut, sgain, mp, scp, x = filter_stepped

    m, sc = tornadox.kalman.smoother_step_sqrt(
        m=m,
        sc=sc,
        m_fut=m_fut,
        sc_fut=sc_fut,
        sgain=sgain,
        mp=mp,
        sq=sq,
        x=x,
    )
    return m, sc


def test_smoother_step_traditional(smoother_stepped_traditional):
    m, sc = smoother_stepped_traditional
    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)


def test_smoother_step_sqrt(smoother_stepped_sqrt):
    m, sc = smoother_stepped_sqrt
    assert isinstance(m, jnp.ndarray)
    assert isinstance(sc, jnp.ndarray)


def test_smoother_step_values(smoother_stepped_sqrt, smoother_stepped_traditional):
    m1, sc1 = smoother_stepped_sqrt
    m2, sc2 = smoother_stepped_traditional
    assert jnp.allclose(m1, m2)
    assert jnp.allclose(sc1, sc2)
