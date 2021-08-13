"""Tests for ODESolver interfaces."""


import dataclasses

import jax.numpy as jnp

import tornado


@dataclasses.dataclass
class EulerState:
    ivp: tornado.ivp.InitialValueProblem
    y: jnp.array
    t: float
    error_estimate: jnp.array
    reference_state: jnp.array


class EulerAsODESolver(tornado.odesolver.ODESolver):
    def initialize(self, ivp):
        return EulerState(
            ivp=ivp, y=ivp.y0, t=ivp.t0, error_estimate=None, reference_state=ivp.y0
        )

    def attempt_step(self, state, dt):
        y = state.y + dt * state.ivp.f(state.t, state.y)
        t = state.t + dt
        return EulerState(
            ivp=state.ivp, y=y, t=t, error_estimate=None, reference_state=y
        )


def test_odesolver():
    constant_steps = tornado.step.ConstantSteps(dt=0.1)
    solver_order = 2
    solver = EulerAsODESolver(steprule=constant_steps, solver_order=solver_order)
    assert isinstance(solver, tornado.odesolver.ODESolver)

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=1.5)
    gen_sol = solver.solution_generator(ivp)
    for idx, _ in enumerate(gen_sol):
        pass
    assert idx > 0

    gen_sol = solver.solution_generator(ivp, stop_at=jnp.array([1.234]))
    ts = jnp.array([state.t for state in gen_sol])
    assert jnp.isin(1.234, ts)
