"""Tests for the EK1 implementation."""

import dataclasses

import jax
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


# Tests for full solves.


# Handy abbreviation for the long parametrize decorator
EK1_VERSIONS = [
    tornado.ek1.ReferenceEK1,
    tornado.ek1.DiagonalEK1,
    tornado.ek1.TruncatedEK1,
]
all_ek1_versions = pytest.mark.parametrize("ek1_version", EK1_VERSIONS)


@all_ek1_versions
def test_full_solve_compare_scipy(
    ek1_version, ivp, steps, scipy_solution, num_derivatives
):
    """Assert the ODEFilter solves an ODE appropriately."""
    final_t_scipy, final_y_scipy = scipy_solution

    ek1 = ek1_version(num_derivatives=num_derivatives, ode_dimension=2, steprule=steps)
    sol_gen = ek1.solution_generator(ivp=ivp)
    for state in sol_gen:
        if state.t > ivp.t0:
            pass

    final_t_ek1 = state.t
    if isinstance(ek1, tornado.ek1.ReferenceEK1):
        final_y_ek1 = ek1.P0 @ state.y.mean
    else:
        final_y_ek1 = state.y.mean[0]
    assert jnp.allclose(final_t_scipy, final_t_ek1)
    assert jnp.allclose(final_y_scipy, final_y_ek1, rtol=1e-3, atol=1e-3)


# Fixtures for tests for initialize, attempt_step, etc.


# Handy selection of test parametrizations
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
def test_init_type(approx_initialized):
    _, init_approx = approx_initialized
    assert isinstance(init_approx.y, tornado.rv.BatchedMultivariateNormal)


@all_ek1_approximations
def test_approx_ek1_initialize_values(approx_initialized, d, n):
    init_ref, init_approx = approx_initialized
    full_cov_as_batch = full_cov_as_batched_cov(
        init_ref.y.cov, expected_shape=init_approx.y.cov.shape
    )
    assert jnp.allclose(init_approx.t, init_ref.t)
    assert jnp.allclose(init_approx.y.mean, init_ref.y.mean.reshape((n, d), order="F"))
    assert jnp.allclose(init_approx.y.cov_sqrtm, full_cov_as_batch)
    assert jnp.allclose(init_approx.y.cov, full_cov_as_batch)


@all_ek1_approximations
def test_approx_ek1_initialize_cov_type(approx_initialized):
    _, init_approx = approx_initialized

    assert isinstance(init_approx.y.cov_sqrtm, jnp.ndarray)
    assert isinstance(init_approx.y.cov, jnp.ndarray)


# Tests for attempt_step (common for all approximations)


@all_ek1_approximations
def test_attempt_step_type(approx_stepped):
    _, step_approx = approx_stepped
    assert isinstance(step_approx.y, tornado.rv.BatchedMultivariateNormal)


@all_ek1_approximations
def test_approx_ek1_attempt_step_y_shapes(approx_stepped, ivp, num_derivatives):
    step_ref, step_approx = approx_stepped
    d, n = ivp.dimension, num_derivatives + 1

    assert step_approx.y.mean.shape == (n, d)
    assert step_approx.y.cov_sqrtm.shape == (d, n, n)
    assert step_approx.y.cov.shape == (d, n, n)


@all_ek1_approximations
def test_approx_ek1_attempt_step_y_cov_type(approx_stepped):
    _, step_approx = approx_stepped
    assert isinstance(step_approx.y.cov_sqrtm, jnp.ndarray)
    assert isinstance(step_approx.y.cov, jnp.ndarray)


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
    ref_cov_as_batch = full_cov_as_batched_cov(
        step_ref.y.cov, expected_shape=step_approx.y.cov.shape
    )

    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    assert jnp.allclose(step_approx.y.cov.todense(), ref_cov_as_batch)


@only_ek1_truncated
def test_approx_ek1_attempt_step_y_values(approx_stepped):
    step_ref, step_approx = approx_stepped

    num_blocks = step_approx.y.cov.array_stack.shape[0]
    block_shape = step_approx.y.cov.array_stack.shape[1:3]
    ref_cov_as_batch = tornado.linops.truncate_block_diagonal(
        step_ref.y.cov,
        num_blocks=num_blocks,
        block_shape=block_shape,
    )

    assert jnp.allclose(step_approx.y.mean, step_ref.y.mean)
    assert jnp.allclose(step_approx.y.cov, ref_cov_as_batch, rtol=5e-4, atol=5e-4)


# Tests for lower-level functions (only types and shapes, not values)
# Common fixtures: mean, covariance, 1d-system-matrices, 1d-preconditioner


@pytest.fixture
def n():
    return 5


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
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def phi_1d(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def sq_1d(n):
    return jnp.arange(1, 1 + n ** 2).reshape((n, n))


@pytest.fixture
def p_1d(n):
    return jnp.diag(jnp.arange(n))


@pytest.fixture
def t(ivp):
    return ivp.t0 + 0.123456


@pytest.fixture
def f(ivp):
    return ivp.f


@pytest.fixture
def df(ivp):
    return ivp.df


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

    # ODE fixtures: jacobians, residuals

    @staticmethod
    @pytest.fixture
    def J(e0, m, ivp):
        return ivp.df(0.0, e0 @ m)

    @staticmethod
    @pytest.fixture
    def h(e0, e1, J):
        return e1 - J @ e0

    @staticmethod
    @pytest.fixture
    def z(h, m):
        return h @ m

    # Test functions

    @staticmethod
    def test_predict_mean(m, phi, n, d):
        mp = tornado.ek1.ReferenceEK1.predict_mean(m, phi)
        assert mp.shape == (n * d,)

    @staticmethod
    def test_predict_cov_sqrtm(sc, phi, sq, n, d):
        scp = tornado.ek1.ReferenceEK1.predict_cov_sqrtm(sc, phi, sq)
        assert scp.shape == (n * d, n * d)

    @staticmethod
    @pytest.fixture
    def evaluated(ivp, m, e0, e1, p, t, f, df):
        return tornado.ek1.ReferenceEK1.evaluate_ode(
            t=t,
            f=f,
            df=df,
            p=p,
            m_pred=m,
            e0=e0,
            e1=e1,
        )

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
        return tornado.ek1.ReferenceEK1.estimate_error(h, sq, z)

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


class TestLowLevelDiagonalEK1Functions:
    """Test suite for low-level, diagonal EK1 functions."""

    @staticmethod
    @pytest.fixture
    def sc_as_bd(sc_1d, d):
        return jnp.stack([sc_1d] * d)

    @staticmethod
    @pytest.fixture
    def sq_as_bd(sq_1d, d):
        return jnp.stack([sq_1d] * d)

    @staticmethod
    @pytest.fixture
    def evaluated(t, f, df, p_1d, m_as_matrix):
        return tornado.ek1.DiagonalEK1.evaluate_ode(
            t=t, f=f, df=df, p_1d=p_1d, m_pred=m_as_matrix
        )

    @staticmethod
    @pytest.fixture
    def J(evaluated):
        _, J, _ = evaluated
        return J

    @staticmethod
    @pytest.fixture
    def z(evaluated):
        _, _, z = evaluated
        return z

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
        assert Jx.shape == (d,)
        assert z.shape == (d,)

    @staticmethod
    def test_predict_mean(m_as_matrix, phi_1d, n, d):
        mp = tornado.ek1.DiagonalEK1.predict_mean(m_as_matrix, phi_1d)
        assert mp.shape == (n, d)

    @staticmethod
    def test_predict_cov_sqrtm(sc_as_bd, phi_1d, sq_as_bd, n, d):
        scp = tornado.ek1.DiagonalEK1.predict_cov_sqrtm(
            sc_bd=sc_as_bd, phi_1d=phi_1d, sq_bd=sq_as_bd
        )
        assert scp.shape == (d, n, n)

    @staticmethod
    @pytest.fixture
    def diagonal_ek1_error_estimated(p_1d, J, sq_as_bd, z):
        return tornado.ek1.DiagonalEK1.estimate_error(
            p_1d=p_1d, J=J, sq_bd=sq_as_bd, z=z
        )

    @staticmethod
    def test_calibrate(diagonal_ek1_error_estimated):
        _, sigma = diagonal_ek1_error_estimated
        assert sigma.shape == ()
        assert sigma >= 0.0

    @staticmethod
    def test_error_estimate(diagonal_ek1_error_estimated, d):
        error_estimate, _ = diagonal_ek1_error_estimated
        assert error_estimate.shape == (d,)
        assert jnp.all(error_estimate >= 0.0)

    @staticmethod
    @pytest.fixture
    def observed(J, p_1d, sc_as_bd):
        return tornado.ek1.DiagonalEK1.observe_cov_sqrtm(
            p_1d=p_1d,
            J=J,
            sc_bd=sc_as_bd,
        )

    @staticmethod
    def test_observe_cov_sqrtm(observed, d, n):
        ss, kgain = observed
        assert ss.shape == (d,)
        assert kgain.shape == (d, n, 1)

    @staticmethod
    def test_correct_cov_sqrtm(J, p_1d, observed, sc_as_bd, d, n):
        _, kgain = observed
        new_sc = tornado.ek1.DiagonalEK1.correct_cov_sqrtm(
            p_1d=p_1d,
            J=J,
            sc_bd=sc_as_bd,
            kgain=kgain,
        )
        assert new_sc.shape == (d, n, n)

    @staticmethod
    def test_correct_mean(m_as_matrix, observed, z, d, n):
        _, kgain = observed
        new_mean = tornado.ek1.DiagonalEK1.correct_mean(m=m_as_matrix, kgain=kgain, z=z)
        assert new_mean.shape == (n, d)


# Auxiliary functions


def full_cov_as_batched_cov(cov, expected_shape):
    """Auxiliary function to make tests more convenient."""
    n, m, k = expected_shape
    return tornado.linops.truncate_block_diagonal(cov, num_blocks=n, block_shape=(m, k))
