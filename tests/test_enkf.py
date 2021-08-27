import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tornadox


@pytest.fixture
def ivp():
    return tornadox.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)


@pytest.fixture
def d(ivp):
    return ivp.y0.shape[0]


@pytest.fixture
def steps():
    dt = 0.1
    return tornadox.step.ConstantSteps(dt)


@pytest.fixture
def num_derivatives():
    return 2


@pytest.fixture
def ensemble_size():
    return 100


@pytest.fixture
def ek1_solution(ek1_version, num_derivatives, ivp, steps, ensemble_size):
    ek1 = ek1_version(
        num_derivatives=num_derivatives, steprule=steps, ensemble_size=ensemble_size
    )
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        if state.t > ivp.t0:
            pass

    final_t_ek1 = state.t
    if isinstance(ek1, tornadox.ek1.ReferenceEK1):
        final_y_ek1 = ek1.P0 @ state.y.mean
    else:
        final_y_ek1 = ek1.P0 @ state.mean.reshape(-1)
    return final_t_ek1, final_y_ek1


@pytest.fixture
def scipy_solution(ivp):
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]
    return final_t_scipy, final_y_scipy


# Tests for full solves.


# Handy abbreviation for the long parametrize decorator
EK1_VERSIONS = [
    tornadox.ek1.ReferenceEK1,
    tornadox.enkf.EnK1,
]
all_ek1_versions = pytest.mark.parametrize("ek1_version", EK1_VERSIONS)


@all_ek1_versions
def test_full_solve_compare_scipy(ek1_solution, scipy_solution):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution
    final_t_ek1, final_y_ek1 = ek1_solution

    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)
