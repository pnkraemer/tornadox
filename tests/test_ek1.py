"""Tests for the EK1 implementation."""

import dataclasses

import jax.numpy as jnp
from scipy.integrate import solve_ivp

import tornado


def test_reference_ek1_constant_steps():
    """Assert the reference solver returns a similar solution to SciPy.

    As long as this test passes, we can test the more efficient solvers against this one here.
    """

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)
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


def test_diagonal_ek1_constant_steps():
    # only "constant steps", because there is no error estimation yet.
    old_ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)

    # Diagonal Jacobian
    new_df = lambda t, y: jnp.diag(jnp.diag(old_ivp.df(t, y)))
    ivp = tornado.ivp.InitialValueProblem(
        f=old_ivp.f,
        df=new_df,
        t0=old_ivp.t0,
        tmax=old_ivp.tmax,
        y0=old_ivp.y0,
    )

    steps = tornado.step.ConstantSteps(0.1)
    reference_ek1 = tornado.ek1.ReferenceEK1(
        num_derivatives=4, ode_dimension=2, steprule=steps
    )
    diagonal_ek1 = tornado.ek1.DiagonalEK1(
        num_derivatives=4, ode_dimension=2, steprule=steps
    )

    # Initialize works as expected
    init_ref = reference_ek1.initialize(ivp=ivp)
    init_diag = diagonal_ek1.initialize(ivp=ivp)
    assert jnp.allclose(init_diag.t, init_ref.t)
    assert jnp.allclose(init_diag.y.mean, init_ref.y.mean)
    assert isinstance(init_diag.y.cov_cholesky, tornado.linops.BlockDiagonal)
    assert jnp.allclose(init_diag.y.cov_cholesky.todense(), init_ref.y.cov_cholesky)

    # Attempt step works as expected
    step_ref = reference_ek1.attempt_step(state=init_ref, dt=0.12345)
    step_diag = diagonal_ek1.attempt_step(state=init_diag, dt=0.12345)
    assert jnp.allclose(init_diag.t, init_ref.t)
    assert jnp.allclose(step_diag.y.mean, step_ref.y.mean)
    assert isinstance(step_diag.y.cov_cholesky, tornado.linops.BlockDiagonal)
    assert jnp.allclose(
        (step_diag.y.cov_cholesky @ step_diag.y.cov_cholesky.T).todense(),
        step_ref.y.cov_cholesky @ step_ref.y.cov_cholesky.T,
    )
    assert jnp.all(jnp.diag(step_diag.y.cov_cholesky.todense()) >= 0.0)


def test_diagonal_ek1_adaptive_steps():
    # only "constant steps", because there is no error estimation yet.
    old_ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.5, stiffness_constant=1.0)

    # Diagonal Jacobian
    new_df = lambda t, y: jnp.diag(jnp.diag(old_ivp.df(t, y)))
    ivp = tornado.ivp.InitialValueProblem(
        f=old_ivp.f,
        df=new_df,
        t0=old_ivp.t0,
        tmax=old_ivp.tmax,
        y0=old_ivp.y0,
    )

    steps = tornado.step.ConstantSteps(0.1)
    diagonal_ek1 = tornado.ek1.DiagonalEK1(
        num_derivatives=4, ode_dimension=2, steprule=steps
    )
    init_diag = diagonal_ek1.initialize(ivp=ivp)
    assert isinstance(init_diag.y.cov_cholesky, tornado.linops.BlockDiagonal)

    # Attempt step works as expected
    d = diagonal_ek1.iwp.wiener_process_dimension
    n = diagonal_ek1.iwp.num_derivatives
    step_diag = diagonal_ek1.attempt_step(state=init_diag, dt=0.12345)
    assert isinstance(step_diag.y.cov_cholesky, tornado.linops.BlockDiagonal)
    assert isinstance(step_diag.y.mean, jnp.ndarray)
    assert isinstance(step_diag.reference_state, jnp.ndarray)
    assert isinstance(step_diag.error_estimate, jnp.ndarray)
    assert step_diag.y.mean.shape == (d * (n + 1),)
    assert step_diag.reference_state.shape == (d,)
    assert step_diag.error_estimate.shape == (d,)
    assert jnp.all(step_diag.reference_state >= 0)


def test_diagonal_ek1_adaptive_steps_full_solve():

    ivp = tornado.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]

    dt = jnp.mean(jnp.diff(scipy_sol.t))
    steps = tornado.step.AdaptiveSteps(first_dt=dt, abstol=1e-3, reltol=1e-3)
    ek1 = tornado.ek1.DiagonalEK1(num_derivatives=4, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek1 = state.t
    final_y_ek1 = ek1.P0 @ state.y.mean
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)
