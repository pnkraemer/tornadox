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
    tornadox.ek1.ReferenceEK1,
    tornadox.ek1.DiagonalEK1,
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
        tornadox.ek1.DiagonalEK1,
    ],
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


class TestLowLevelReferenceEK1Functions:
    """Test suite for the low-level EK1 functions"""

    # Common fixtures: full system matrices, projection matrices

    @staticmethod
    @pytest.fixture
    def phi(phi_1d, d):
        return jnp.kron(jnp.eye(d), phi_1d)

    @staticmethod
    @pytest.fixture
    def e0(n, d):
        e0_1d = jnp.eye(1, n).reshape((-1,))
        return jnp.kron(jnp.eye(d), e0_1d)

    @staticmethod
    @pytest.fixture
    def e1(n, d):
        # e_{-1} as a dummy for e1 -- only the shapes matter anyway
        e1_1d = jnp.flip(jnp.eye(1, n)).reshape((-1,))
        return jnp.kron(jnp.eye(d), e1_1d)

    @staticmethod
    @pytest.fixture
    def sq(sq_1d, d):
        return jnp.kron(jnp.eye(d), sq_1d)

    @staticmethod
    @pytest.fixture
    def sc(sc_1d, d):
        return jnp.kron(jnp.eye(d), sc_1d)

    @staticmethod
    @pytest.fixture
    def p(p_1d, d):
        return jnp.kron(jnp.eye(d), p_1d)

    # ODE fixtures (Jacobians, residuals, linearisations)

    @staticmethod
    @pytest.fixture
    def evaluated(ivp, m, e0, e1, p, t, f, df):
        return tornadox.ek1.ReferenceEK1.evaluate_ode(
            t=t,
            f=f,
            df=df,
            p=p,
            m_pred=m,
            e0=e0,
            e1=e1,
        )

    @staticmethod
    @pytest.fixture
    def h(evaluated):
        h, _ = evaluated
        return h

    @staticmethod
    @pytest.fixture
    def z(evaluated):
        _, z = evaluated
        return z

    # Test functions

    @staticmethod
    def test_predict_mean(m, phi, n, d):
        mp = tornadox.ek1.ReferenceEK1.predict_mean(m, phi)
        assert mp.shape == (n * d,)

    @staticmethod
    def test_predict_cov_sqrtm(sc, phi, sq, n, d):
        scp = tornadox.ek1.ReferenceEK1.predict_cov_sqrtm(sc, phi, sq)
        assert scp.shape == (n * d, n * d)

    @staticmethod
    def test_evaluate_ode_type(evaluated):
        h, z = evaluated
        assert isinstance(h, jnp.ndarray)
        assert isinstance(z, jnp.ndarray)

    @staticmethod
    def test_evaluate_ode_shape(evaluated, d, n):
        h, z = evaluated
        assert h.shape == (d, d * n)
        assert z.shape == (d,)

    @staticmethod
    @pytest.fixture
    def reference_ek1_error_estimated(h, sq, z):
        return tornadox.ek1.ReferenceEK1.estimate_error(h, sq, z)

    @staticmethod
    def test_calibrate(reference_ek1_error_estimated):
        _, sigma = reference_ek1_error_estimated
        assert sigma.shape == ()
        assert sigma >= 0.0

    @staticmethod
    def test_error_estimate(reference_ek1_error_estimated, d):
        error_estimate, _ = reference_ek1_error_estimated
        assert error_estimate.shape == (d,)
        assert jnp.all(error_estimate >= 0.0)


# Batched versions of 1d system matrices


@pytest.fixture
def sc_as_bd(sc_1d, d):
    return jnp.stack([sc_1d] * d)


@pytest.fixture
def sq_as_bd(sq_1d, d):
    return jnp.stack([sq_1d] * d)


class TestLowLevelBatchedEK1Functions:
    @staticmethod
    def test_predict_mean(m_as_matrix, phi_1d, n, d):
        mp = tornadox.ek1.BatchedEK1.predict_mean(m_as_matrix, phi_1d)
        assert mp.shape == (n, d)

    @staticmethod
    def test_predict_cov_sqrtm(sc_as_bd, phi_1d, sq_as_bd, n, d):
        scp = tornadox.ek1.BatchedEK1.predict_cov_sqrtm(
            sc_bd=sc_as_bd, phi_1d=phi_1d, sq_bd=sq_as_bd
        )
        assert scp.shape == (d, n, n)


class TestLowLevelDiagonalEK1Functions:
    """Test suite for low-level, diagonal EK1 functions."""

    # ODE fixtures

    @staticmethod
    @pytest.fixture
    def evaluated(t, f, df_diagonal, p_1d_raw, m_as_matrix):
        return tornadox.ek1.DiagonalEK1.evaluate_ode(
            t=t, f=f, df_diagonal=df_diagonal, p_1d_raw=p_1d_raw, m_pred=m_as_matrix
        )

    @staticmethod
    @pytest.fixture
    def Jx_diagonal(evaluated):
        _, Jx_diagonal, _ = evaluated
        return Jx_diagonal

    @staticmethod
    @pytest.fixture
    def z(evaluated):
        _, _, z = evaluated
        return z

    # Tests for the low-level functions

    @staticmethod
    def test_evaluate_ode_type(evaluated):
        fx, Jx_diagonal, z = evaluated
        assert isinstance(fx, jnp.ndarray)
        assert isinstance(Jx_diagonal, jnp.ndarray)
        assert isinstance(z, jnp.ndarray)

    @staticmethod
    def test_evaluate_ode_shape(evaluated, d):
        fx, Jx_diagonal, z = evaluated
        assert fx.shape == (d,)
        assert Jx_diagonal.shape == (d,)
        assert z.shape == (d,)

    @staticmethod
    @pytest.fixture
    def diagonal_ek1_error_estimated(p_1d_raw, Jx_diagonal, sq_as_bd, z):
        return tornadox.ek1.DiagonalEK1.estimate_error(
            p_1d_raw=p_1d_raw, Jx_diagonal=Jx_diagonal, sq_bd=sq_as_bd, z=z
        )

    @staticmethod
    def test_calibrate(diagonal_ek1_error_estimated, d):
        _, sigma = diagonal_ek1_error_estimated
        assert sigma.shape == (d,)
        assert jnp.all(sigma >= 0.0)

    @staticmethod
    def test_error_estimate(diagonal_ek1_error_estimated, d):
        error_estimate, _ = diagonal_ek1_error_estimated
        assert error_estimate.shape == (d,)
        assert jnp.all(error_estimate >= 0.0)

    @staticmethod
    @pytest.fixture
    def observed(Jx_diagonal, p_1d_raw, sc_as_bd):
        return tornadox.ek1.DiagonalEK1.observe_cov_sqrtm(
            p_1d_raw=p_1d_raw,
            Jx_diagonal=Jx_diagonal,
            sc_bd=sc_as_bd,
        )

    @staticmethod
    def test_observe_cov_sqrtm(observed, d, n):
        ss, kgain = observed
        assert ss.shape == (d,)
        assert kgain.shape == (d, n, 1)

    @staticmethod
    def test_correct_cov_sqrtm(Jx_diagonal, p_1d_raw, observed, sc_as_bd, d, n):
        _, kgain = observed
        new_sc = tornadox.ek1.DiagonalEK1.correct_cov_sqrtm(
            p_1d_raw=p_1d_raw,
            Jx_diagonal=Jx_diagonal,
            sc_bd=sc_as_bd,
            kgain=kgain,
        )
        assert new_sc.shape == (d, n, n)

    @staticmethod
    def test_correct_mean(m_as_matrix, observed, z, d, n):
        _, kgain = observed
        new_mean = tornadox.ek1.DiagonalEK1.correct_mean(
            m=m_as_matrix, kgain=kgain, z=z
        )
        assert new_mean.shape == (n, d)


# Auxiliary functions


def full_cov_as_batched_cov(cov, expected_shape):
    """Auxiliary function to make tests more convenient."""
    n, m, k = expected_shape
    return tornadox.experimental.linops.truncate_block_diagonal(
        cov, num_blocks=n, block_shape=(m, k)
    )
