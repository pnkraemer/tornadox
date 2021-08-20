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


@pytest.fixture
def scipy_solution(ivp):
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]
    return final_t_scipy, final_y_scipy


@pytest.fixture
def num_derivatives():
    return 4


# Tests for reference EK1


# Tests for full solve.
all_ek1_versions = pytest.mark.parametrize(
    "ek1_version",
    [tornado.ek1.ReferenceEK1, tornado.ek1.DiagonalEK1, tornado.ek1.TruncatedEK1],
)


@all_ek1_versions
def test_full_solve_compare_scipy(
    ek1_version, ivp, steps, scipy_solution, num_derivatives
):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution

    ek1 = ek1_version(num_derivatives=num_derivatives, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        pass

    final_t_ek1 = state.t
    final_y_ek1 = ek1.P0 @ state.y.mean
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)


# Fixtures for tests for initialize, attempt_step, etc.

all_ek1_approximations = pytest.mark.parametrize(
    "approx_solver", [tornado.ek1.DiagonalEK1, tornado.ek1.TruncatedEK1]
)
only_ek1_diagonal = pytest.mark.parametrize("approx_solver", [tornado.ek1.DiagonalEK1])
only_ek1_truncated = pytest.mark.parametrize(
    "approx_solver", [tornado.ek1.TruncatedEK1]
)


@pytest.fixture
def solver_triple(ivp, steps, num_derivatives, approx_solver):
    """Assemble a combination of a to-be-tested-EK1 and a ReferenceEK1 with matching parameters."""

    # Diagonal Jacobian into the IVP to make the reference EK1 acknowledge it too.
    if approx_solver == tornado.ek1.DiagonalEK1:
        old_ivp = ivp
        new_df = lambda t, y: jnp.diag(jnp.diag(old_ivp.df(t, y)))
        ivp = tornado.ivp.InitialValueProblem(
            f=old_ivp.f,
            df=new_df,
            t0=old_ivp.t0,
            tmax=old_ivp.tmax,
            y0=old_ivp.y0,
        )

    d, n = ivp.dimension, num_derivatives
    reference_ek1 = tornado.ek1.ReferenceEK1(
        num_derivatives=n, ode_dimension=d, steprule=steps
    )
    ek1_approx = approx_solver(num_derivatives=n, ode_dimension=d, steprule=steps)

    return ek1_approx, reference_ek1, ivp


@pytest.fixture
def approx_initialized(solver_triple):
    """Initialize the to-be-tested EK1 and the reference EK1."""

    ek1_approx, reference_ek1, ivp = solver_triple

    init_ref = reference_ek1.initialize(ivp=ivp)
    init_approx = ek1_approx.initialize(ivp=ivp)

    return init_ref, init_approx


@pytest.fixture
def approx_stepped(solver_triple, approx_initialized):
    """Attempt a step with the to-be-tested-EK1 and the reference EK1."""

    ek1_approx, reference_ek1, _ = solver_triple
    init_ref, init_approx = approx_initialized

    step_ref = reference_ek1.attempt_step(state=init_ref, dt=0.12345)
    step_approx = ek1_approx.attempt_step(state=init_approx, dt=0.12345)

    return step_ref, step_approx


# Tests for initialization


@all_ek1_approximations
def test_approx_ek1_initialize_values(approx_initialized):
    init_ref, init_approx = approx_initialized

    assert jnp.allclose(init_approx.t, init_ref.t)
    assert jnp.allclose(init_approx.y.mean, init_ref.y.mean)
    assert jnp.allclose(init_approx.y.cov_sqrtm.todense(), init_ref.y.cov_sqrtm)
    assert jnp.allclose(init_approx.y.cov.todense(), init_ref.y.cov)


@all_ek1_approximations
def test_approx_ek1_initialize_cov_type(approx_initialized):
    _, init_approx = approx_initialized

    assert isinstance(init_approx.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    assert isinstance(init_approx.y.cov, tornado.linops.BlockDiagonal)


# Tests for attempt_step (common for all approximations)


@all_ek1_approximations
def test_approx_ek1_attempt_step_y_shapes(approx_stepped, ivp, num_derivatives):
    step_ref, step_approx = approx_stepped
    d, n = ivp.dimension, num_derivatives

    assert step_approx.y.mean.shape == (d * (n + 1),)
    assert step_approx.y.cov_sqrtm.todense().shape == step_ref.y.cov_sqrtm.shape
    assert step_approx.y.cov.todense().shape == step_ref.y.cov.shape


@all_ek1_approximations
def test_approx_ek1_attempt_step_y_cov_type(approx_stepped):
    _, step_approx = approx_stepped
    assert isinstance(step_approx.y.cov_sqrtm, tornado.linops.BlockDiagonal)
    assert isinstance(step_approx.y.cov, tornado.linops.BlockDiagonal)


@all_ek1_approximations
def test_approx_ek1_attempt_step_error_estimate(approx_stepped, ivp):
    _, step_approx = approx_stepped

    assert isinstance(step_approx.error_estimate, jnp.ndarray)
    assert step_approx.error_estimate.shape == (ivp.dimension,)
    assert jnp.all(step_approx.error_estimate >= 0)


@all_ek1_approximations
def test_approx_ek1_attempt_step_reference_state(approx_stepped, ivp, num_derivatives):
    _, step_approx = approx_stepped

    assert isinstance(step_approx.reference_state, jnp.ndarray)
    assert step_approx.reference_state.shape == (ivp.dimension,)
    assert jnp.all(step_approx.reference_state >= 0)


# Tests for attempt_step (specific to some approximations)


@only_ek1_diagonal
def test_approx_ek1_attempt_step_y_values(approx_stepped):
    step_ref, step_approx = approx_stepped
    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    assert jnp.allclose(step_approx.y.cov.todense(), step_ref.y.cov)


@only_ek1_truncated
def test_approx_ek1_attempt_step_y_values(approx_stepped):
    step_ref, step_approx = approx_stepped

    num_blocks = step_approx.y.cov.array_stack.shape[0]
    block_shape = step_approx.y.cov.array_stack.shape[1:3]
    ref_cov_as_bd_array_stack = tornado.linops.truncate_block_diagonal(
        step_ref.y.cov,
        num_blocks=num_blocks,
        block_shape=block_shape,
    )
    truncated_ref_cov = tornado.linops.BlockDiagonal(
        ref_cov_as_bd_array_stack
    ).todense()

    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    assert jnp.allclose(
        step_approx.y.cov.todense(), truncated_ref_cov, rtol=5e-4, atol=5e-4
    )
