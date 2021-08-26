"""Tests for ODEFilter interfaces."""


import dataclasses

import jax.numpy as jnp
import pytest

import tornado


@dataclasses.dataclass
class EulerState:
    ivp: tornado.ivp.InitialValueProblem
    y: jnp.array
    t: float
    error_estimate: jnp.array
    reference_state: jnp.array


class EulerAsODEFilter(tornado.odefilter.ODEFilter):
    def initialize(self, ivp):
        y = tornado.rv.MultivariateNormal(
            ivp.y0, cov_sqrtm=jnp.zeros((ivp.y0.shape[0], ivp.y0.shape[0]))
        )
        return EulerState(
            ivp=ivp, y=y, t=ivp.t0, error_estimate=None, reference_state=ivp.y0
        )

    def attempt_step(self, state, dt):
        y = state.y.mean + dt * state.ivp.f(state.t, state.y.mean)
        t = state.t + dt
        y = tornado.rv.MultivariateNormal(
            y, cov_sqrtm=jnp.zeros((y.shape[0], y.shape[0]))
        )
        return EulerState(
            ivp=state.ivp, y=y, t=t, error_estimate=None, reference_state=y
        )


@pytest.fixture
def ivp():
    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=1.5)
    return ivp


@pytest.fixture
def steps():
    return tornado.step.ConstantSteps(dt=0.1)


@pytest.fixture
def solver(steps):
    solver_order = 1
    solver = EulerAsODEFilter(
        steprule=steps,
        num_derivatives=solver_order,
    )
    return solver


def test_simulate_final_point(ivp, solver):
    sol = solver.simulate_final_state(ivp)
    assert isinstance(sol, EulerState)


def test_solve(ivp, solver):
    sol = solver.solve(ivp)
    assert isinstance(sol, tornado.odefilter.ODESolution)

    #
    # gen_sol = solver.solution_generator(ivp)
    # for idx, _ in enumerate(gen_sol):
    #     pass
    # assert idx > 0
    #
    # gen_sol = solver.solution_generator(ivp, stop_at=jnp.array([1.234]))
    # ts = jnp.array([state.t for state in gen_sol])
    # assert jnp.isin(1.234, ts)
