"""Tests for the EK1 implementation."""

import dataclasses

import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

import tornado


# Commonly reused fixtures
@pytest.fixture
def ivp():
    return tornado.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)


@pytest.fixture
def steps():
    dt = 0.1
    return tornado.step.AdaptiveSteps(first_dt=dt, abstol=1e-3, reltol=1e-3)


# Tests for reference EK1


def test_full_solve_reference_ek0_compare_scipy(ivp, steps):
    """Assert the EK0 solves an ODE correctly.

    This makes the EK0 a valid reference to
    check the optimised implementations against below.
    """
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]

    ek1 = tornado.ek1.ReferenceEK1(num_derivatives=4, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek1 = state.t
    final_y_ek1 = ek1.P0 @ state.y.mean
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)


# Tests for diagonal EK1


def test_diagonal_ek1_attempt_step(ivp, steps):
    old_ivp = ivp
    # Diagonal Jacobian
    new_df = lambda t, y: jnp.diag(jnp.diag(old_ivp.df(t, y)))
    ivp = tornado.ivp.InitialValueProblem(
        f=old_ivp.f,
        df=new_df,
        t0=old_ivp.t0,
        tmax=old_ivp.tmax,
        y0=old_ivp.y0,
    )

    d, n = 2, 4
    reference_ek1 = tornado.ek1.ReferenceEK1(
        num_derivatives=n, ode_dimension=d, steprule=steps
    )
    diagonal_ek1 = tornado.ek1.DiagonalEK1(
        num_derivatives=n, ode_dimension=d, steprule=steps
    )

    # Initialize works as expected
    init_ref = reference_ek1.initialize(ivp=ivp)
    init_diag = diagonal_ek1.initialize(ivp=ivp)
    assert jnp.allclose(init_diag.t, init_ref.t)
    assert jnp.allclose(init_diag.y.mean, init_ref.y.mean)
    assert isinstance(init_diag.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    assert jnp.allclose(init_diag.y.cov_sqrtm.todense(), init_ref.y.cov_sqrtm)

    # Attempt step works as expected
    step_ref = reference_ek1.attempt_step(state=init_ref, dt=0.12345)
    step_diag = diagonal_ek1.attempt_step(state=init_diag, dt=0.12345)
    assert jnp.allclose(init_diag.t, init_ref.t)
    assert jnp.allclose(step_diag.y.mean, step_ref.y.mean)
    assert isinstance(step_diag.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    received = (step_diag.y.cov_sqrtm @ step_diag.y.cov_sqrtm.T).todense()
    expected = step_ref.y.cov_sqrtm @ step_ref.y.cov_sqrtm.T
    assert received.shape == expected.shape
    assert jnp.allclose(received, expected), received - expected
    assert isinstance(step_diag.reference_state, jnp.ndarray)
    assert isinstance(step_diag.error_estimate, jnp.ndarray)
    assert step_diag.y.mean.shape == (d * (n + 1),)
    assert step_diag.reference_state.shape == (d,)
    assert step_diag.error_estimate.shape == (d,)
    assert jnp.all(step_diag.reference_state >= 0)


def test_diagonal_ek1_adaptive_steps_full_solve(ivp, steps):

    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]

    ek1 = tornado.ek1.DiagonalEK1(num_derivatives=4, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek1 = state.t
    final_y_ek1 = ek1.P0 @ state.y.mean
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)


# Tests for truncated EK1 aka EK1


def test_truncated_ek1_attempt_step(ivp, steps):

    d, n = 2, 4
    reference_ek1 = tornado.ek1.ReferenceEK1(
        num_derivatives=n, ode_dimension=d, steprule=steps
    )
    truncated_ek1 = tornado.ek1.TruncatedEK1(
        num_derivatives=n, ode_dimension=d, steprule=steps
    )

    # Initialize works as expected
    init_ref = reference_ek1.initialize(ivp=ivp)
    init_trunc = truncated_ek1.initialize(ivp=ivp)
    assert jnp.allclose(init_trunc.t, init_ref.t)
    assert jnp.allclose(init_trunc.y.mean, init_ref.y.mean)
    assert isinstance(init_trunc.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    assert jnp.allclose(init_trunc.y.cov_sqrtm.todense(), init_ref.y.cov_sqrtm)

    # Attempt step works as expected
    step_ref = reference_ek1.attempt_step(state=init_ref, dt=0.12345)
    step_trunc = truncated_ek1.attempt_step(state=init_trunc, dt=0.12345)

    assert jnp.allclose(step_trunc.t, step_ref.t)
    assert jnp.allclose(step_trunc.y.mean, step_ref.y.mean)
    assert isinstance(step_trunc.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    received = step_trunc.y.cov
    expected_dense = step_ref.y.cov
    expected_as_bd_array_stack = tornado.linops.truncate_block_diagonal(
        expected_dense,
        num_blocks=step_trunc.y.cov_sqrtm.array_stack.shape[0],
        block_shape=step_trunc.y.cov_sqrtm.array_stack.shape[1:3],
    )
    expected = tornado.linops.BlockDiagonal(expected_as_bd_array_stack)
    # Dont relax the tolerance here -- it is sharp!
    assert jnp.allclose(
        received.array_stack, expected.array_stack, rtol=5e-4, atol=5e-4
    )
    assert received.todense().shape == expected.todense().shape

    # check the usual reference state and error estimation stuff
    assert isinstance(step_trunc.reference_state, jnp.ndarray)
    assert isinstance(step_trunc.error_estimate, jnp.ndarray)
    assert step_trunc.y.mean.shape == (d * (n + 1),)
    assert step_trunc.reference_state.shape == (d,)
    assert step_trunc.error_estimate.shape == (d,)
    assert jnp.all(step_trunc.reference_state >= 0)
