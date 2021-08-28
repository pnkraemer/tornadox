"""Tests for initial value problems and examples thereof."""

import jax
import jax.numpy as jnp
import pytest

import tornadox

IVPs = [
    tornadox.ivp.vanderpol(),
    tornadox.ivp.brusselator(),
    tornadox.ivp.lorenz96(),
    tornadox.ivp.lorenz96_loop(),
]


class TestIVPCommonBehaviour:
    @staticmethod
    @pytest.mark.parametrize("ivp", IVPs)
    def test_type(ivp):
        assert isinstance(ivp, tornadox.ivp.InitialValueProblem)

    @staticmethod
    @pytest.mark.parametrize("ivp", IVPs)
    def test_f(ivp):
        f = ivp.f(ivp.t0, ivp.y0)
        assert isinstance(f, jnp.ndarray)
        assert f.shape == ivp.y0.shape

    @staticmethod
    @pytest.mark.parametrize("ivp", IVPs)
    def test_df(ivp):
        if ivp.df is not None:
            df = ivp.df(ivp.t0, ivp.y0)
            assert isinstance(df, jnp.ndarray)
            assert df.shape == (ivp.y0.shape[0], ivp.y0.shape[0])

    @staticmethod
    @pytest.mark.parametrize("ivp", IVPs)
    def test_jittable(ivp):
        def fun(*problem):
            f, t0, tmax, y0, df = problem
            return t0, tmax, y0

        fun_jitted = jax.jit(fun, static_argnums=(0, 4))
        out = tornadox.ivp.InitialValueProblem(ivp.f, *fun_jitted(*ivp), ivp.df)
        assert type(out) == type(ivp)


class TestLorenzFormulations:
    """Both Lorenz versions should have identical behaviour."""

    @staticmethod
    @pytest.fixture
    def both_lorenz_formulations():
        lorenz = tornadox.ivp.lorenz96()
        lorenz_loop = tornadox.ivp.lorenz96_loop()
        return lorenz, lorenz_loop

    @staticmethod
    def test_same_t0(both_lorenz_formulations):
        lorenz, lorenz_loop = both_lorenz_formulations
        assert lorenz.t0 == lorenz_loop.t0

    @staticmethod
    def test_same_tmax(both_lorenz_formulations):
        lorenz, lorenz_loop = both_lorenz_formulations
        assert lorenz.tmax == lorenz_loop.tmax

    @staticmethod
    def test_same_y0(both_lorenz_formulations):
        lorenz, lorenz_loop = both_lorenz_formulations

        assert jnp.allclose(lorenz.y0, lorenz_loop.y0)

    @staticmethod
    def test_same_f(both_lorenz_formulations):
        lorenz, lorenz_loop = both_lorenz_formulations
        fx = lorenz.f(lorenz.t0, lorenz.y0)
        fx_loop = lorenz_loop.f(lorenz_loop.t0, lorenz_loop.y0)
        assert jnp.allclose(fx, fx_loop)

    @staticmethod
    def test_same_df(both_lorenz_formulations):
        lorenz, lorenz_loop = both_lorenz_formulations
        dfx = lorenz.df(lorenz.t0, lorenz.y0)
        dfx_loop = lorenz_loop.df(lorenz_loop.t0, lorenz_loop.y0)
        assert jnp.allclose(dfx, dfx_loop)
