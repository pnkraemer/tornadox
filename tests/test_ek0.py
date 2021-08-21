import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

import tornado


@pytest.fixture
def ivp():
    return tornado.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)


@pytest.fixture
def steps():
    dt = 0.1
    return tornado.step.AdaptiveSteps(first_dt=dt, abstol=1e-3, reltol=1e-3)


@pytest.fixture
def num_derivatives():
    return 4


@pytest.fixture
def ek0_solution(ek0_version, num_derivatives, ivp, steps):
    ek0 = ek0_version(num_derivatives=num_derivatives, ode_dimension=2, steprule=steps)
    sol_gen = ek0.solution_generator(ivp=ivp)
    for state in sol_gen:
        if state.t > ivp.t0:
            pass

    final_t_ek0 = state.t
    # if isinstance(ek0, tornado.ek0.ReferenceEK0):
    #     final_y_ek0 = ek0.P0 @ state.y.mean
    # else:
    #     final_y_ek0 = state.y.mean[0]
    final_y_ek0 = ek0.P0 @ state.y.mean
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
    tornado.ek0.KroneckerEK0,
]
all_ek0_versions = pytest.mark.parametrize("ek0_version", EK0_VERSIONS)


@all_ek0_versions
def test_full_solve_compare_scipy(ek0_solution, scipy_solution):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution
    final_t_ek0, final_y_ek0 = ek0_solution

    assert jnp.allclose(final_t_scipy, final_t_ek0)
    assert jnp.allclose(final_y_scipy, final_y_ek0, rtol=1e-3, atol=1e-3)


#
#
#
# def test_vec_trick_mul_full():
#     for i in range(100):
#         d1, d2, d3, d4 = np.random.randint(1, 10, size=4)
#         K1, K2 = np.random.rand(d1, d2), np.random.rand(d3, d4)
#         v = np.random.rand(d2 * d4)
#         assert np.allclose(vec_trick_mul_full(K1, K2, v), np.kron(K1, K2) @ v)
#
#
# def test_vec_trick_mul_right():
#     for i in range(100):
#         d = np.random.randint(1, 10)
#         q = np.random.randint(1, 10)
#         K2 = np.random.rand(q + 1, q + 1)
#         v = np.random.rand(d * (q + 1))
#         assert np.allclose(vec_trick_mul_right(K2, v), np.kron(np.eye(d), K2) @ v)
#
#
# @pytest.fixture
# def ivp():
#     return tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)
#
#
# @pytest.fixture
# def scipy_solution(ivp):
#     return solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0, rtol=1e-8, atol=1e-8)
#
#
# def test_reference_ek0_constant_steps(ivp, scipy_solution):
#     scipy_final_t = scipy_solution.t[-1]
#     scipy_final_y = scipy_solution.y[:, -1]
#
#     constant_steps = tornado.step.ConstantSteps(0.01)
#     solver = ReferenceEK0(
#         ode_dimension=ivp.dimension, steprule=constant_steps, num_derivatives=4
#     )
#     for state in solver.solution_generator(ivp=ivp):
#         pass
#
#     assert jnp.allclose(scipy_final_t, state.t)
#     assert jnp.allclose(scipy_final_y, solver.P0 @ state.y.mean, rtol=1e-3, atol=1e-3)
#
#
# def test_ek0_constant_steps(ivp, scipy_solution):
#     scipy_final_t = scipy_solution.t[-1]
#     scipy_final_y = scipy_solution.y[:, -1]
#
#     constant_steps = tornado.step.ConstantSteps(0.01)
#     solver = KroneckerEK0(
#         ode_dimension=ivp.dimension, steprule=constant_steps, num_derivatives=4
#     )
#     for state in solver.solution_generator(ivp=ivp):
#         pass
#
#     assert jnp.allclose(scipy_final_t, state.t)
#     assert jnp.allclose(scipy_final_y, solver.P0 @ state.y.mean, rtol=1e-3, atol=1e-3)
#
#
# def test_ek0_adaptive_steps(ivp, scipy_solution):
#     scipy_final_t = scipy_solution.t[-1]
#     scipy_final_y = scipy_solution.y[:, -1]
#
#     srule = tornado.step.AdaptiveSteps(first_dt=0.01, abstol=1e-6, reltol=1e-3)
#     solver = KroneckerEK0(
#         ode_dimension=ivp.dimension, steprule=srule, num_derivatives=4
#     )
#     for state in solver.solution_generator(ivp=ivp):
#         pass
#
#     assert jnp.allclose(scipy_final_t, state.t)
#     assert jnp.allclose(scipy_final_y, solver.P0 @ state.y.mean, rtol=1e-3, atol=1e-3)
