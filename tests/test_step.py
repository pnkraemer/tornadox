"""Tests for stepsize selection."""

import jax.numpy as jnp
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


