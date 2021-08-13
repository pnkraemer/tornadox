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


def test_adaptive_steps():
    abstol = 0.1
    reltol = 0.01
    steprule = tornado.step.AdaptiveSteps(first_dt=0.1, abstol=abstol, reltol=reltol)
    assert isinstance(steprule, tornado.step.AdaptiveSteps)

    # < 1 accepted, > 1 rejected
    assert steprule.is_accepted(scaled_error_estimate=0.99)
    assert not steprule.is_accepted(scaled_error_estimate=1.01)

    # Accepting a step makes the next one larger
    assert (
        steprule.suggest(
            previous_dt=0.3, scaled_error_estimate=0.5, local_convergence_rate=2
        )
        > 0.3
    )
    assert (
        steprule.suggest(
            previous_dt=0.3, scaled_error_estimate=2.0, local_convergence_rate=2
        )
        < 0.3
    )

    # Error estimation to scaled norm 1d
    unscaled_error_estimate = jnp.array([0.5])
    reference_state = jnp.array([2.0])
    E = steprule.scale_error_estimate(
        unscaled_error_estimate=unscaled_error_estimate, reference_state=reference_state
    )
    scaled_error = unscaled_error_estimate / (abstol + reltol * reference_state)
    assert jnp.allclose(E, scaled_error)

    # Error estimation to scaled norm 2d
    unscaled_error_estimate = jnp.array([0.5, 0.6])
    reference_state = jnp.array([2.0, 3.0])
    E = steprule.scale_error_estimate(
        unscaled_error_estimate=unscaled_error_estimate, reference_state=reference_state
    )
    scaled_error = jnp.linalg.norm(
        unscaled_error_estimate / (abstol + reltol * reference_state)
    ) / jnp.sqrt(2)
    assert jnp.allclose(E, scaled_error)

    # min_step exception
    steprule.min_step = 0.1
    with pytest.raises(ValueError):
        steprule.suggest(
            previous_dt=1e-1,
            scaled_error_estimate=1_000_000_000,
            local_convergence_rate=1,
        )

    # max_step exception
    steprule.max_step = 10.0
    with pytest.raises(ValueError):
        steprule.suggest(
            previous_dt=9.0,
            scaled_error_estimate=1 / 1_000_000_000,
            local_convergence_rate=1,
        )
