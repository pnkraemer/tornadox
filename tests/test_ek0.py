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


# Test fixtures for attempt_step and initialize


@pytest.fixture
def solver_tuple(steps, num_derivatives, d):
    reference_ek0 = tornado.ek0.ReferenceEK0(
        num_derivatives=num_derivatives, ode_dimension=d, steprule=steps
    )
    kronecker_ek0 = tornado.ek0.KroneckerEK0(
        num_derivatives=num_derivatives, ode_dimension=d, steprule=steps
    )

    return kronecker_ek0, reference_ek0


@pytest.fixture
def initialized_both(solver_tuple, ivp):
    kronecker_ek0, reference_ek0 = solver_tuple

    kronecker_init = kronecker_ek0.initialize(ivp=ivp)
    reference_init = reference_ek0.initialize(ivp=ivp)

    return kronecker_init, reference_init


@pytest.fixture
def stepped_both(solver_tuple, ivp, initialized_both):

    kronecker_ek0, reference_ek0 = solver_tuple
    kronecker_init, reference_init = initialized_both

    kronecker_stepped = kronecker_ek0.attempt_step(state=kronecker_init, dt=0.12345)
    reference_stepped = reference_ek0.attempt_step(state=reference_init, dt=0.12345)

    return kronecker_stepped, reference_stepped


# Tests for initialize


def test_init_type(initialized_both):
    kronecker_init, _ = initialized_both
    assert isinstance(kronecker_init.y, tornado.rv.MultivariateNormal)


def test_init_values(initialized_both, d):
    kronecker_init, reference_init = initialized_both

    kron_cov_sqrtm = jnp.kron(jnp.eye(d), kronecker_init.y.cov_sqrtm)
    kron_cov = jnp.kron(jnp.eye(d), kronecker_init.y.cov)
    assert jnp.allclose(kronecker_init.t, reference_init.t)
    assert jnp.allclose(kronecker_init.y.mean, reference_init.y.mean)
    assert jnp.allclose(kron_cov_sqrtm, reference_init.y.cov_sqrtm)
    assert jnp.allclose(kron_cov, reference_init.y.cov)


def test_init_shape_kronecker(initialized_both, d, num_derivatives):
    kronecker_init, _ = initialized_both

    # shorthand
    n = num_derivatives + 1
    y = kronecker_init.y
    m, sc, c = y.mean, y.cov_sqrtm, y.cov

    assert m.shape == (d * n,)
    assert sc.shape == (n, n)
    assert c.shape == (n, n)


def test_init_shape_reference(initialized_both, d, num_derivatives):
    _, reference_init = initialized_both

    # shorthand
    n = num_derivatives + 1
    y = reference_init.y
    m, sc, c = y.mean, y.cov_sqrtm, y.cov

    assert m.shape == (d * n,)
    assert sc.shape == (d * n, d * n)
    assert c.shape == (d * n, d * n)


# Tests for each attempt step


@pytest.fixture
def stepped_kronecker(stepped_both):
    stepped_kron, _ = stepped_both
    return stepped_kron


@pytest.fixture
def stepped_reference(stepped_both):
    _, stepped_reference = stepped_both
    return stepped_reference


# Test for shapes of output


def test_attempt_step_y_shapes_kronecker(stepped_kronecker, d, num_derivatives):
    n = num_derivatives + 1
    assert stepped_kronecker.y.mean.shape == (n * d,)
    assert stepped_kronecker.y.cov_sqrtm.shape == (n, n)
    assert stepped_kronecker.y.cov.shape == (n, n)


def test_attempt_step_y_shapes_reference(stepped_reference, d, num_derivatives):
    n = num_derivatives + 1
    assert stepped_reference.y.mean.shape == (n * d,)
    assert stepped_reference.y.cov_sqrtm.shape == (d * n, d * n)
    assert stepped_reference.y.cov.shape == (d * n, d * n)


def test_attempt_step_error_estimate_kronecker(stepped_kronecker, d):

    assert isinstance(stepped_kronecker.error_estimate, jnp.ndarray)
    assert stepped_kronecker.error_estimate.shape == (d,)
    assert jnp.all(stepped_kronecker.error_estimate >= 0)


def test_attempt_step_error_estimate_reference(stepped_reference, d):

    assert isinstance(stepped_reference.error_estimate, jnp.ndarray)
    assert stepped_reference.error_estimate.shape == (d,)
    assert jnp.all(stepped_reference.error_estimate >= 0)


def test_attempt_step_reference_state_kronecker(stepped_kronecker, d):

    assert isinstance(stepped_kronecker.reference_state, jnp.ndarray)
    assert stepped_kronecker.reference_state.shape == (d,)
    assert jnp.all(stepped_kronecker.reference_state >= 0)


def test_attempt_step_reference_state_reference(stepped_reference, d):

    assert isinstance(stepped_reference.reference_state, jnp.ndarray)
    assert stepped_reference.reference_state.shape == (d,)
    assert jnp.all(stepped_reference.reference_state >= 0)


# Test for values of output


def test_attempt_step_values_y_mean(stepped_kronecker, stepped_reference):
    m1, m2 = stepped_reference.y.mean, stepped_kronecker.y.mean
    assert jnp.allclose(m1, m2)


def test_attempt_step_values_y_cov(stepped_kronecker, stepped_reference, d):
    c1, c2_small = stepped_reference.y.cov, stepped_kronecker.y.cov
    c2 = jnp.kron(jnp.eye(d), c2_small)
    assert jnp.allclose(c1, c2)


def test_attempt_step_values_y_error_estimate(stepped_kronecker, stepped_reference, d):
    e1, e2 = stepped_reference.error_estimate, stepped_kronecker.error_estimate
    assert jnp.allclose(e1, e2)


def test_attempt_step_values_y_reference_state(stepped_kronecker, stepped_reference, d):
    r1, r2 = stepped_reference.reference_state, stepped_kronecker.reference_state
    assert jnp.allclose(r1, r2)


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
