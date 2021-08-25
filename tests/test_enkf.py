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


@pytest.fixture
def solver_tuple(steps, num_derivatives, d, ensemble_size):
    reference_ek0 = tornado.ek0.ReferenceEK0(
        num_derivatives=num_derivatives, steprule=steps
    )
    enk_0 = tornado.enkf.EnK0(
        num_derivatives=num_derivatives, steprule=steps, ensemble_size=ensemble_size
    )

    return enk_0, reference_ek0


@pytest.fixture
def initialized_both(solver_tuple, ivp):
    enk_0, reference_ek0 = solver_tuple

    enk_0_init = enk_0.initialize(ivp=ivp)
    reference_init = reference_ek0.initialize(ivp=ivp)

    return enk_0_init, reference_init


@pytest.fixture
def stepped_both(solver_tuple, ivp, initialized_both):

    enk_0, reference_ek0 = solver_tuple
    enk_0_init, reference_init = initialized_both

    enk_0_stepped = enk_0.attempt_step(ensemble=enk_0_init, dt=0.12345)
    reference_stepped = reference_ek0.attempt_step(state=reference_init, dt=0.12345)

    return enk_0_stepped, reference_stepped


# Tests for initialize


def test_init_type(initialized_both):
    enk_0_init, _ = initialized_both
    assert isinstance(enk_0_init.samples, jnp.ndarray)


def test_init_values(initialized_both, d):
    enk_0_init, reference_init = initialized_both

    enk0_mean = enk_0_init.mean.reshape((-1,), order="F")
    enk0_cov_sqrtm = enk_0_init.cov_sqrtm
    enk0_cov = enk_0_init.sample_cov
    assert jnp.allclose(enk_0_init.t, reference_init.t)
    assert jnp.allclose(enk0_mean, reference_init.y.mean, rtol=1e-3, atol=1e-3)


def test_init_shape_enk_0(initialized_both, d, num_derivatives):
    enk_0_init, _ = initialized_both

    # shorthand
    n = num_derivatives + 1
    m = enk_0_init.mean
    cov = enk_0_init.sample_cov
    covL = enk_0_init.cov_sqrtm

    assert m.shape == (n * d, 1)
    assert cov.shape == (n * d, n * d)
    assert covL.shape == (n * d, n * d)


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
def stepped_enk_0(stepped_both):
    stepped_kron, _ = stepped_both
    return stepped_kron


@pytest.fixture
def stepped_reference(stepped_both):
    _, stepped_reference = stepped_both
    return stepped_reference


# Test for shapes of output


def test_attempt_step_y_type(stepped_enk_0):
    assert isinstance(stepped_enk_0, tornado.enkf.StateEnsemble)


def test_attempt_step_y_shapes_enk_0(stepped_enk_0, d, num_derivatives):
    n = num_derivatives + 1
    m = stepped_enk_0.mean
    cov = stepped_enk_0.sample_cov
    covL = stepped_enk_0.cov_sqrtm

    assert m.shape == (n * d, 1)
    assert cov.shape == (n * d, n * d)
    assert covL.shape == (n * d, n * d)


def test_attempt_step_y_shapes_reference(stepped_reference, d, num_derivatives):
    n = num_derivatives + 1
    assert stepped_reference.y.mean.shape == (n * d,)
    assert stepped_reference.y.cov_sqrtm.shape == (d * n, d * n)
    assert stepped_reference.y.cov.shape == (d * n, d * n)


# Test for values of output


def test_attempt_step_values_y_mean(stepped_enk_0, stepped_reference):
    m1, m2 = stepped_reference.y.mean, stepped_enk_0.mean
    assert jnp.allclose(m1, m2.reshape((-1,), order="F"), rtol=1e-2, atol=1e-2)


def test_attempt_step_values_y_cov(stepped_enk_0, stepped_reference, d):
    c1, c2 = stepped_reference.y.cov, stepped_enk_0.sample_cov
    assert jnp.allclose(c1, c2, rtol=1e-2, atol=1e-2)
