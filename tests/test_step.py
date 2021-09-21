"""Tests for stepsize selection."""

import jax.numpy as jnp
import pytest

import tornadox


def test_propose_first_dt():

    ivp = tornadox.ivp.vanderpol()

    dt = tornadox.step.propose_first_dt(ivp.f, ivp.t0, ivp.y0)
    assert dt > 0


@pytest.fixture
def ivp():
    return tornadox.ivp.vanderpol()


class TestConstantSteps:
    @staticmethod
    @pytest.fixture
    def dt():
        return 0.1

    @staticmethod
    @pytest.fixture
    def steprule(dt):
        steprule = tornadox.step.ConstantSteps(dt)
        return steprule

    @staticmethod
    def test_propose_is_dt(steprule, dt):
        proposed = steprule.suggest(previous_dt=jnp.nan, scaled_error_estimate=0.1)
        assert proposed == dt

    @staticmethod
    def test_always_accept(steprule):
        assert steprule.is_accepted(scaled_error_estimate=0.1)

    @staticmethod
    def test_error_estimate_is_none(steprule):

        # "None" does not matter here, these quantities are not used.
        assert steprule.scale_error_estimate(None, None) is None

    @staticmethod
    def test_first_dt_is_dt(steprule, ivp, dt):
        first_dt = steprule.first_dt(*ivp)
        assert first_dt == dt


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
        steprule = tornadox.step.AdaptiveSteps(abstol=abstol, reltol=reltol)
        return steprule

    @staticmethod
    def test_type(steprule):
        assert isinstance(steprule, tornadox.step.AdaptiveSteps)

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
    def test_first_dt(steprule, ivp):
        dt = steprule.first_dt(*ivp)
        assert dt > 0.0
