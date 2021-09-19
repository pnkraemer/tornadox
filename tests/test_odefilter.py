"""Tests for ODEFilter interfaces."""

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import tornadox


class EulerState(namedtuple("_EulerState", "t y error_estimate reference_state")):
    pass


class EulerAsODEFilter(tornadox.odefilter.ODEFilter):
    def initialize(self, f, t0, tmax, y0, df, df_diagonal):
        y = tornadox.rv.MultivariateNormal(
            y0, cov_sqrtm=jnp.zeros((y0.shape[0], y0.shape[0]))
        )
        return EulerState(y=y, t=t0, error_estimate=jnp.nan * y0, reference_state=y0)

    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):
        y = state.y.mean + dt * f(state.t, state.y.mean)
        t = state.t + dt
        y = tornadox.rv.MultivariateNormal(
            y, cov_sqrtm=jnp.zeros((y.shape[0], y.shape[0]))
        )
        new_state = EulerState(
            y=y, t=t, error_estimate=jnp.nan * y0, reference_state=y.mean
        )
        return new_state, {}


@pytest.fixture
def ivp():
    ivp = tornadox.ivp.vanderpol(t0=0.0, tmax=1.5)
    return ivp


@pytest.fixture
def steps():
    return tornadox.step.ConstantSteps(dt=0.1)


@pytest.fixture
def solver(steps):
    solver_order = 1
    solver = EulerAsODEFilter(
        steprule=steps,
        num_derivatives=solver_order,
    )
    return solver


def test_simulate_final_point(ivp, solver):
    sol, _ = solver.simulate_final_state(ivp)
    assert isinstance(sol, EulerState)


def test_solve(ivp, solver):
    sol = solver.solve(ivp)
    assert isinstance(sol, tornadox.odefilter.ODESolution)


@pytest.fixture
def locations():
    return jnp.array([1.234])


def test_solve_stop_at(ivp, solver, locations):
    sol = solver.solve(ivp, stop_at=locations)
    assert jnp.isin(locations[0], jnp.array(sol.t))


def test_odefilter_state_jittable(ivp):
    def fun(state):
        t, y, err, ref = state
        return tornadox.odefilter.ODEFilterState(t, y, err, ref)

    fun_jitted = jax.jit(fun)
    x = jnp.zeros(3)
    state = tornadox.odefilter.ODEFilterState(
        t=0, y=x, error_estimate=x, reference_state=x
    )
    out = fun_jitted(state)
    assert type(out) == type(state)
