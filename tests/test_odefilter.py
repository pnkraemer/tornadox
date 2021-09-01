"""Tests for ODEFilter interfaces."""


import dataclasses

import jax.numpy as jnp
import pytest

import tornadox


@dataclasses.dataclass
class EulerState:
    ivp: tornadox.ivp.InitialValueProblem
    y: jnp.array
    t: float
    error_estimate: jnp.array
    reference_state: jnp.array


class EulerAsODEFilter(tornadox.odefilter.ODEFilter):
    def initialize(self, ivp):
        y = tornadox.rv.MultivariateNormal(
            ivp.y0, cov_sqrtm=jnp.zeros((ivp.y0.shape[0], ivp.y0.shape[0]))
        )
        return EulerState(
            ivp=ivp, y=y, t=ivp.t0, error_estimate=None, reference_state=ivp.y0
        )

    def attempt_step(self, state, dt):
        y = state.y.mean + dt * state.ivp.f(state.t, state.y.mean)
        t = state.t + dt
        y = tornadox.rv.MultivariateNormal(
            y, cov_sqrtm=jnp.zeros((y.shape[0], y.shape[0]))
        )
        new_state = EulerState(
            ivp=state.ivp, y=y, t=t, error_estimate=None, reference_state=y
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
