import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tornado


@pytest.fixture
def ivp():
    return tornado.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)


@pytest.fixture
def d(ivp):
    return ivp.y0.shape[0]


@pytest.fixture
def steps():
    dt = 0.1
    return tornado.step.ConstantSteps(dt)


@pytest.fixture
def num_derivatives():
    return 2


@pytest.fixture
def ensemble_size():
    return 100


@pytest.fixture
def ek0_solution(ek0_version, num_derivatives, ivp, steps, ensemble_size):
    ek0 = ek0_version(
        num_derivatives=num_derivatives, steprule=steps, ensemble_size=ensemble_size
    )
    sol_gen = ek0.solution_generator(ivp=ivp)
    for state in sol_gen:
        if state.t > ivp.t0:
            pass

    final_t_ek0 = state.t
    if isinstance(ek0, tornado.ek0.ReferenceEK0):
        final_y_ek0 = ek0.P0 @ state.y.mean
    else:
        final_y_ek0 = ek0.P0 @ state.mean.reshape(-1)
    return final_t_ek0, final_y_ek0


@pytest.fixture
def scipy_solution(ivp):
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]
    return final_t_scipy, final_y_scipy


# Tests for full solves.


# Handy abbreviation for the long parametrize decorator
EK0_VERSIONS = [
    tornado.ek0.ReferenceEK0,
    tornado.enkf.EnK0,
]
all_ek0_versions = pytest.mark.parametrize("ek0_version", EK0_VERSIONS)


@all_ek0_versions
def test_full_solve_compare_scipy(ek0_solution, scipy_solution):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution
    final_t_ek0, final_y_ek0 = ek0_solution

    assert jnp.allclose(final_t_scipy, final_t_ek0)
    assert jnp.allclose(final_y_scipy, final_y_ek0, rtol=1e-3, atol=1e-3)


# Test fixtures for attempt_step and initialize
