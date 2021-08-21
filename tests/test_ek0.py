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
    return 2


@pytest.fixture
def ek0_solution(ek0_version, num_derivatives, ivp, steps):
    ek0 = ek0_version(num_derivatives=num_derivatives, ode_dimension=2, steprule=steps)
    sol_gen = ek0.solution_generator(ivp=ivp)
    for state in sol_gen:
        if state.t > ivp.t0:
            pass

    final_t_ek0 = state.t
    if isinstance(ek0, tornado.ek0.ReferenceEK0):
        final_y_ek0 = ek0.P0 @ state.y.mean
    else:
        final_y_ek0 = state.y.mean[0]
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
    assert isinstance(kronecker_init.y, tornado.rv.MatrixNormal)


def test_init_values(initialized_both, d):
    kronecker_init, reference_init = initialized_both

    kron_mean = kronecker_init.y.mean.reshape((-1,), order="F")
    kron_cov_sqrtm = kronecker_init.y.dense_cov_sqrtm()
    kron_cov = kronecker_init.y.dense_cov()
    assert jnp.allclose(kronecker_init.t, reference_init.t)
    assert jnp.allclose(kron_mean, reference_init.y.mean)
    assert jnp.allclose(kron_cov_sqrtm, reference_init.y.cov_sqrtm)
    assert jnp.allclose(kron_cov, reference_init.y.cov)


def test_init_shape_kronecker(initialized_both, d, num_derivatives):
    kronecker_init, _ = initialized_both

    # shorthand
    n = num_derivatives + 1
    y = kronecker_init.y
    m = y.mean
    sc1, sc2 = y.cov_sqrtm_1, y.cov_sqrtm_2
    c1, c2 = y.cov_1, y.cov_2

    assert m.shape == (n, d)
    assert sc2.shape == (n, n)
    assert sc1.shape == (d, d)
    assert c1.shape == (d, d)
    assert c2.shape == (n, n)


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


def test_attempt_step_y_type(stepped_kronecker):
    assert isinstance(stepped_kronecker.y, tornado.rv.MatrixNormal)


def test_attempt_step_y_shapes_kronecker(stepped_kronecker, d, num_derivatives):
    n = num_derivatives + 1
    y = stepped_kronecker.y
    m = y.mean
    sc1, sc2 = y.cov_sqrtm_1, y.cov_sqrtm_2
    c1, c2 = y.cov_1, y.cov_2

    assert m.shape == (n, d)
    assert sc1.shape == (d, d)
    assert sc2.shape == (n, n)
    assert c1.shape == (d, d)
    assert c2.shape == (n, n)


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
    assert jnp.allclose(m1, m2.reshape((-1,), order="F"))


def test_attempt_step_values_y_cov(stepped_kronecker, stepped_reference, d):
    c1, c2 = stepped_reference.y.cov, stepped_kronecker.y.dense_cov()
    assert jnp.allclose(c1, c2)


def test_attempt_step_values_y_error_estimate(stepped_kronecker, stepped_reference, d):
    e1, e2 = stepped_reference.error_estimate, stepped_kronecker.error_estimate
    assert jnp.allclose(e1, e2)


def test_attempt_step_values_y_reference_state(stepped_kronecker, stepped_reference, d):
    r1, r2 = stepped_reference.reference_state, stepped_kronecker.reference_state
    assert jnp.allclose(r1, r2)
