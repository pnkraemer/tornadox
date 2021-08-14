"""Tests for the EK1 implementation."""

import jax.numpy as jnp
from scipy.integrate import solve_ivp

import tornado


def test_reference_ek1():
    """Assert the solver returns a similar solution to SciPy."""

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]

    dt = jnp.mean(jnp.diff(scipy_sol.t))

    steps = tornado.step.ConstantSteps(dt)
    ek1 = tornado.ek1.ReferenceEK1(num_derivatives=4, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek1 = state.t
    final_y_ek1 = ek1.P0 @ state.y.mean
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)
