"""Tests for stepsize selection."""

import jax.numpy as jnp
import pytest

import tornado


def test_propose_first_dt():

    ivp = tornado.ivp.vanderpol()

    dt = tornado.step.propose_first_dt(ivp)
    assert dt > 0


def test_constant_steps():
    dt = 0.1
    steprule = tornado.step.ConstantSteps(dt)

    proposed = steprule.suggest(previous_dt=jnp.nan, scaled_error_estimate=0.1)
    assert proposed == 0.1
    assert steprule.is_accepted(scaled_error_estimate=0.1)

    # "None" does not matter here, these quantities are not used.
    assert jnp.isnan(steprule.scale_error_estimate(None, None))


class TestAdaptiveSteps:
    @staticmethod
    @pytest.fixture
    def abstol():
        return 0.1

    @staticmethod
    @pytest.fixture
    def reltol():
        return 0.01

    @staticmethod
    @pytest.fixture
    def steprule(abstol, reltol):
        steprule = tornado.step.AdaptiveSteps(
            first_dt=0.1, abstol=abstol, reltol=reltol
        )
        return steprule

    @staticmethod
    def test_type(steprule):
        assert isinstance(steprule, tornado.step.AdaptiveSteps)

    @staticmethod
    def test_accept_less_than_1(steprule):
        assert steprule.is_accepted(scaled_error_estimate=0.99)

    @staticmethod
    def test_reject_more_than_1(steprule):
        assert not steprule.is_accepted(scaled_error_estimate=1.01)

    @staticmethod
    def test_accepting_makes_next_step_larger(steprule):
        assert (
            steprule.suggest(
                previous_dt=0.3, scaled_error_estimate=0.5, local_convergence_rate=2
            )
            > 0.3
        )

    @staticmethod
    def test_rejecting_makes_next_step_smaller(steprule):
        assert (
            steprule.suggest(
                previous_dt=0.3, scaled_error_estimate=2.0, local_convergence_rate=2
            )
            < 0.3
        )

    @staticmethod
    def test_scale_error_estimate_1d(steprule, abstol, reltol):
        unscaled_error_estimate = jnp.array([0.5])
        reference_state = jnp.array([2.0])
        E = steprule.scale_error_estimate(
            unscaled_error_estimate=unscaled_error_estimate,
            reference_state=reference_state,
        )
        scaled_error = unscaled_error_estimate / (abstol + reltol * reference_state)
        assert jnp.allclose(E, scaled_error)

    @staticmethod
    def test_scale_error_estimate_2d(steprule, abstol, reltol):
        unscaled_error_estimate = jnp.array([0.5, 0.6])
        reference_state = jnp.array([2.0, 3.0])
        E = steprule.scale_error_estimate(
            unscaled_error_estimate=unscaled_error_estimate,
            reference_state=reference_state,
        )
        scaled_error = jnp.linalg.norm(
            unscaled_error_estimate / (abstol + reltol * reference_state)
        ) / jnp.sqrt(2)
        assert jnp.allclose(E, scaled_error)

    @staticmethod
    def test_min_step_exception(steprule):
        steprule.min_step = 0.1
        with pytest.raises(ValueError):
            steprule.suggest(
                previous_dt=1e-1,
                scaled_error_estimate=1_000_000_000,
                local_convergence_rate=1,
            )

    @staticmethod
    def test_max_step_exception(steprule):
        steprule.max_step = 10.0
        with pytest.raises(ValueError):
            steprule.suggest(
                previous_dt=9.0,
                scaled_error_estimate=1 / 1_000_000_000,
                local_convergence_rate=1,
            )
