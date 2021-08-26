"""Tests for initial value problems and examples thereof."""

import jax.numpy
import pytest

import tornado

IVPs = [tornado.ivp.vanderpol(), tornado.ivp.brusselator(), tornado.ivp.lorenz96()]


@pytest.mark.parametrize("ivp", IVPs)
def test_ivp(ivp):
    assert isinstance(ivp, tornado.ivp.InitialValueProblem)


@pytest.mark.parametrize("ivp", IVPs)
def test_f(ivp):
    f = ivp.f(ivp.t0, ivp.y0)
    assert isinstance(f, jax.numpy.ndarray)
    assert f.shape == ivp.y0.shape


@pytest.mark.parametrize("ivp", IVPs)
def test_df(ivp):
    if ivp.df is not None:
        df = ivp.df(ivp.t0, ivp.y0)
        assert isinstance(df, jax.numpy.ndarray)
        assert df.shape == (ivp.y0.shape[0], ivp.y0.shape[0])
