"""Tests for the EK1 implementation."""

import dataclasses

import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

import tornadox

# Commonly reused fixtures


@pytest.fixture
def ivp():
    return tornadox.ivp.vanderpol(t0=0.0, tmax=0.25, stiffness_constant=1.0)


@pytest.fixture
def steps():
    return tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-3)


@pytest.fixture
def scipy_solution(ivp):
    scipy_sol = solve_ivp(ivp.f, t_span=(ivp.t0, ivp.tmax), y0=ivp.y0)
    final_t_scipy = scipy_sol.t[-1]
    final_y_scipy = scipy_sol.y[:, -1]
    return final_t_scipy, final_y_scipy


@pytest.fixture
def num_derivatives():
    return 2


# Tests for full solves.


# Handy abbreviation for the long parametrize decorator
EK1_VERSIONS = [
    tornadox.experimental.truncated.EarlyTruncationEK1,
    tornadox.experimental.truncated.TruncationEK1,
]
all_ek1_versions = pytest.mark.parametrize("ek1_version", EK1_VERSIONS)


@all_ek1_versions
def test_full_solve_compare_scipy(
    ek1_version, ivp, steps, scipy_solution, num_derivatives
):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution

    ek1 = ek1_version(num_derivatives=num_derivatives, steprule=steps)
    state, _ = ek1.simulate_final_state(ivp=ivp)

    final_t_ek1 = state.t
    final_y_ek1 = state.y.mean[0]
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)


@all_ek1_versions
def test_info_dict(ek1_version, ivp, num_derivatives):
    """Assert the ODEFilter solves an ODE appropriately."""
    num_steps = 5
    steprule = tornadox.step.ConstantSteps((ivp.tmax - ivp.t0) / num_steps)
    ek1 = ek1_version(num_derivatives=num_derivatives, steprule=steprule)
    _, info = ek1.simulate_final_state(ivp=ivp)
    assert info["num_f_evaluations"] == num_steps
    assert info["num_steps"] == num_steps
    assert info["num_attempted_steps"] == num_steps
    if isinstance(ek1, tornadox.ek1.DiagonalEK1):
        assert info["num_df_diagonal_evaluations"] == num_steps
    else:
        assert info["num_df_evaluations"] == num_steps


# Fixtures for tests for initialize, attempt_step, etc.


# Handy selection of test parametrizations
all_ek1_approximations = pytest.mark.parametrize(
    "approx_solver",
    [
        tornadox.experimental.truncated.EarlyTruncationEK1,
        tornadox.experimental.truncated.TruncationEK1,
    ],
)
only_ek1_truncation = pytest.mark.parametrize(
    "approx_solver", [tornadox.experimental.truncated.TruncationEK1]
)
only_ek1_early_truncation = pytest.mark.parametrize(
    "approx_solver", [tornadox.experimental.truncated.EarlyTruncationEK1]
)


large_and_small_steps = pytest.mark.parametrize("dt", [0.12121, 12.345])


@pytest.fixture
def solver_triple(ivp, steps, num_derivatives, approx_solver):
    """Assemble a combination of a to-be-tested-EK1 and a ReferenceEK1 with matching parameters."""

    # Diagonal Jacobian into the IVP to make the reference EK1 acknowledge it too.
    # This is important, because it allows checking that the outputs of DiagonalEK1 and ReferenceEK1
    # coincide exactly, which confirms correct implementation of the DiagonalEK1.
    # The key step here is to make the Jacobian of the IVP diagonal.
    if approx_solver == tornadox.ek1.DiagonalEK1:
        old_ivp = ivp
        new_df = lambda t, y: jnp.diag(old_ivp.df_diagonal(t, y))
        ivp = tornadox.ivp.InitialValueProblem(
            f=old_ivp.f,
            df=new_df,
            df_diagonal=old_ivp.df_diagonal,
            t0=old_ivp.t0,
            tmax=old_ivp.tmax,
            y0=old_ivp.y0,
        )

    d, n = ivp.dimension, num_derivatives
    reference_ek1 = tornadox.ek1.ReferenceEK1(num_derivatives=n, steprule=steps)
    ek1_approx = approx_solver(num_derivatives=n, steprule=steps)

    return ek1_approx, reference_ek1, ivp


@pytest.fixture
def approx_initialized(solver_triple):
    """Initialize the to-be-tested EK1 and the reference EK1."""

    ek1_approx, reference_ek1, ivp = solver_triple

    init_ref = reference_ek1.initialize(*ivp)
    init_approx = ek1_approx.initialize(*ivp)

    return init_ref, init_approx


@pytest.fixture
def approx_stepped(solver_triple, approx_initialized, dt):
    """Attempt a step with the to-be-tested-EK1 and the reference EK1."""
    ek1_approx, reference_ek1, ivp = solver_triple
    init_ref, init_approx = approx_initialized

    step_ref, _ = reference_ek1.attempt_step(init_ref, dt, *ivp)
    step_approx, _ = ek1_approx.attempt_step(init_approx, dt, *ivp)

    return step_ref, step_approx


# Tests for initialization


@all_ek1_approximations
def test_init_type(approx_initialized):
    _, init_approx = approx_initialized
    assert isinstance(init_approx.y, tornadox.rv.BatchedMultivariateNormal)


@all_ek1_approximations
def test_approx_ek1_initialize_values(approx_initialized, d, n):
    init_ref, init_approx = approx_initialized
    full_cov_as_batch = full_cov_as_batched_cov(
        init_ref.y.cov, expected_shape=init_approx.y.cov.shape
    )
    assert jnp.allclose(init_approx.t, init_ref.t)
    assert jnp.allclose(init_approx.y.mean, init_ref.y.mean)
    assert jnp.allclose(init_approx.y.cov_sqrtm, full_cov_as_batch)
    assert jnp.allclose(init_approx.y.cov, full_cov_as_batch)


@all_ek1_approximations
def test_approx_ek1_initialize_cov_type(approx_initialized):
    _, init_approx = approx_initialized

    assert isinstance(init_approx.y.cov_sqrtm, jnp.ndarray)
    assert isinstance(init_approx.y.cov, jnp.ndarray)


# Tests for attempt_step (common for all approximations)


@large_and_small_steps
@all_ek1_approximations
def test_attempt_step_type(approx_stepped):
    _, step_approx = approx_stepped
    assert isinstance(step_approx.y, tornadox.rv.BatchedMultivariateNormal)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_y_shapes(approx_stepped, ivp, num_derivatives):
    _, step_approx = approx_stepped
    d, n = ivp.dimension, num_derivatives + 1

    assert step_approx.y.mean.shape == (n, d)
    assert step_approx.y.cov_sqrtm.shape == (d, n, n)
    assert step_approx.y.cov.shape == (d, n, n)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_y_types(approx_stepped):
    _, step_approx = approx_stepped
    assert isinstance(step_approx.y.cov_sqrtm, jnp.ndarray)
    assert isinstance(step_approx.y.cov, jnp.ndarray)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_error_estimate_type(approx_stepped, ivp):
    _, step_approx = approx_stepped

    assert isinstance(step_approx.error_estimate, jnp.ndarray)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_error_estimate_shapes(approx_stepped, ivp):
    _, step_approx = approx_stepped
    assert step_approx.error_estimate.shape == (ivp.dimension,)


@large_and_small_steps
@only_ek1_truncation
def test_approx_ek1_attempt_step_error_estimate_values(approx_stepped, ivp):
    step_ref, step_approx = approx_stepped

    assert jnp.all(step_approx.error_estimate >= 0)
    assert jnp.allclose(step_approx.error_estimate, step_ref.error_estimate)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_reference_state_type(
    approx_stepped, ivp, num_derivatives
):
    _, step_approx = approx_stepped

    assert isinstance(step_approx.reference_state, jnp.ndarray)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_reference_state_shape(
    approx_stepped, ivp, num_derivatives
):
    _, step_approx = approx_stepped
    assert step_approx.reference_state.shape == (ivp.dimension,)


@large_and_small_steps
@all_ek1_approximations
def test_approx_ek1_attempt_step_reference_state_value(
    approx_stepped, ivp, num_derivatives
):
    step_ref, step_approx = approx_stepped

    assert jnp.all(step_approx.reference_state >= 0)
    assert jnp.allclose(step_approx.reference_state, step_ref.reference_state)


# Tests for attempt_step (specific to some approximations)


@large_and_small_steps
@only_ek1_truncation
def test_ek1_attempt_step_y_values(approx_stepped):
    step_ref, step_approx = approx_stepped
    ref_cov_as_batch = full_cov_as_batched_cov(
        step_ref.y.cov, expected_shape=step_approx.y.cov.shape
    )
    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    assert jnp.allclose(step_approx.y.cov, ref_cov_as_batch)


@large_and_small_steps
@only_ek1_early_truncation
def test_truncated_ek1_attempt_step_y_values(approx_stepped):
    step_ref, step_approx = approx_stepped

    num_blocks = step_approx.y.cov.shape[0]
    block_shape = step_approx.y.cov.shape[1:3]
    ref_cov_as_batch = tornadox.experimental.linops.truncate_block_diagonal(
        step_ref.y.cov,
        num_blocks=num_blocks,
        block_shape=block_shape,
    )

    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    # The cov approximation is not particularly good, and also step-size dependent.
    # Therefore we do not check values here.


# Tests for lower-level functions (only types and shapes, not values)
# Common fixtures: mean, covariance, 1d-system-matrices, 1d-preconditioner


@pytest.fixture
def n(num_derivatives):
    return num_derivatives + 1


@pytest.fixture
def d(ivp):
    return ivp.dimension


@pytest.fixture
def m(n, d):
    return jnp.arange(1, 1 + n * d) * 1.0


@pytest.fixture
def m_as_matrix(m, n, d):
    return m.reshape((n, d))


@pytest.fixture
def sc_1d(n):
    return jnp.arange(1, 1 + n**2).reshape((n, n))


@pytest.fixture
def phi_1d(n):
    return jnp.arange(1, 1 + n**2).reshape((n, n))


@pytest.fixture
def sq_1d(n):
    return jnp.arange(1, 1 + n**2).reshape((n, n))


@pytest.fixture
def p_1d_raw(n):
    return jnp.arange(1, 1 + n)


@pytest.fixture
def p_1d(p_1d_raw):
    return jnp.diag(p_1d_raw)


# Easy access fixtures for the ODE attributes


@pytest.fixture
def t(ivp):
    return ivp.t0 + 0.123456


@pytest.fixture
def f(ivp):
    return ivp.f


@pytest.fixture
def df(ivp):
    return ivp.df


@pytest.fixture
def df_diagonal(ivp):
    return ivp.df_diagonal


@pytest.fixture
def sc_as_bd(sc_1d, d):
    return jnp.stack([sc_1d] * d)


@pytest.fixture
def sq_as_bd(sq_1d, d):
    return jnp.stack([sq_1d] * d)


class TestLowLevelTruncationEK1Functions:
    """Test suite for low-level, truncated EK1 functions."""

    @staticmethod
    @pytest.fixture
    def evaluated(t, f, df, p_1d_raw, m_as_matrix):
        return tornadox.experimental.truncated.TruncationEK1.evaluate_ode(
            t=t, f=f, df=df, p_1d_raw=p_1d_raw, m_pred=m_as_matrix
        )

    @staticmethod
    @pytest.fixture
    def Jx(evaluated):
        _, Jx, _ = evaluated
        return Jx

    @staticmethod
    @pytest.fixture
    def z(evaluated):
        _, _, z = evaluated
        return z

    # Tests for the low-level functions

    @staticmethod
    def test_evaluate_ode_type(evaluated):
        fx, Jx, z = evaluated
        assert isinstance(fx, jnp.ndarray)
        assert isinstance(Jx, jnp.ndarray)
        assert isinstance(z, jnp.ndarray)

    @staticmethod
    def test_evaluate_ode_shape(evaluated, d):
        fx, Jx, z = evaluated
        assert fx.shape == (d,)
        assert Jx.shape == (d, d)
        assert z.shape == (d,)

    @staticmethod
    @pytest.fixture
    def truncation_ek1_error_estimated(p_1d_raw, Jx, sq_as_bd, z):
        return tornadox.experimental.truncated.TruncationEK1.estimate_error(
            p_1d_raw=p_1d_raw, Jx=Jx, sq_bd=sq_as_bd, z=z
        )

    @staticmethod
    def test_calibrate(truncation_ek1_error_estimated):
        _, sigma = truncation_ek1_error_estimated
        assert sigma.shape == ()
        assert sigma >= 0.0

    @staticmethod
    def test_error_estimate(truncation_ek1_error_estimated, d):
        error_estimate, _ = truncation_ek1_error_estimated
        assert error_estimate.shape == (d,)
        assert jnp.all(error_estimate >= 0.0)

    @staticmethod
    @pytest.fixture
    def observed(Jx, p_1d_raw, sc_as_bd):
        return tornadox.experimental.truncated.TruncationEK1.observe_cov_sqrtm(
            p_1d_raw=p_1d_raw,
            Jx=Jx,
            sc_bd=sc_as_bd,
        )

    @staticmethod
    def test_observe_cov_sqrtm(observed, d, n):
        ss, kgain = observed
        assert ss.shape == (d, d)
        assert kgain.shape == (d * n, d)

    @staticmethod
    def test_correct_mean(m_as_matrix, observed, z, d, n):
        _, kgain = observed
        new_mean = tornadox.experimental.truncated.TruncationEK1.correct_mean(
            m=m_as_matrix, kgain=kgain, z=z
        )
        assert new_mean.shape == (n, d)

    @staticmethod
    def test_correct_cov_sqrtm(Jx, p_1d_raw, observed, sc_as_bd, d, n):
        _, kgain = observed
        new_sc = tornadox.experimental.truncated.TruncationEK1.correct_cov_sqrtm(
            p_1d_raw=p_1d_raw,
            Jx=Jx,
            sc_bd=sc_as_bd,
            kgain=kgain,
        )
        assert new_sc.shape == (d, n, n)


# Auxiliary functions


def full_cov_as_batched_cov(cov, expected_shape):
    """Auxiliary function to make tests more convenient."""
    n, m, k = expected_shape
    return tornadox.experimental.linops.truncate_block_diagonal(
        cov, num_blocks=n, block_shape=(m, k)
    )
