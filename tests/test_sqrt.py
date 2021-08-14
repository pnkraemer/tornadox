"""Tests for square-root utilities."""

import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def iwp():
    return tornado.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1, num_derivatives=1
    )


def test_propagate_cholesky_factor(iwp):
    transition_matrix, process_noise_cholesky = iwp.preconditioned_discretize_1d

    # dummy cholesky factor
    some_chol1 = process_noise_cholesky.copy()
    some_chol2 = process_noise_cholesky.copy()

    # First test: Non-optional S2
    chol = tornado.sqrt.propagate_cholesky_factor(
        S1=(transition_matrix @ some_chol1), S2=some_chol2
    )
    cov = (
        transition_matrix @ some_chol1 @ some_chol1.T @ transition_matrix.T
        + process_noise_cholesky @ process_noise_cholesky.T
    )
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.linalg.cholesky(cov), chol)
    assert jnp.all(jnp.diag(chol) > 0)


def test_propagate_batched_cholesky_factors(iwp):
    transition_matrix, process_noise_cholesky = iwp.preconditioned_discretize_1d
    A = tornado.linops.BlockDiagonal(jnp.stack([transition_matrix] * 3))

    # dummy cholesky factor
    some_chol1 = tornado.linops.BlockDiagonal(
        jnp.stack([process_noise_cholesky.copy()] * 3)
    )
    some_chol2 = tornado.linops.BlockDiagonal(
        jnp.stack([process_noise_cholesky.copy()] * 3)
    )

    # First test: Non-optional S2
    chol = tornado.sqrt.propagate_batched_cholesky_factor(
        (A @ some_chol1).array_stack, some_chol2.array_stack
    )
    chol_as_bd = tornado.linops.BlockDiagonal(chol)
    reference = tornado.sqrt.propagate_cholesky_factor(
        A.todense() @ some_chol1.todense(), some_chol2.todense()
    )
    assert jnp.allclose(chol_as_bd.todense(), reference)

    # # Second test: Optional S2
    # chol = tornado.sqrt.propagate_batched_cholesky_factor(
    #     batched_S1=(A @ some_chol1).array_stack
    # )
    # chol_as_bd = tornado.linops.BlockDiagonal(chol)
    # reference = tornado.sqrt.propagate_cholesky_factor(
    #     A.todense() @ some_chol1.todense()
    # )
    # assert jnp.allclose(chol_as_bd.todense(), reference)
    #


def test_tril_to_positive_tril():
    """Assert that the weird sign(0)=0 behaviour is made up for."""
    matrix = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    result = tornado.sqrt.tril_to_positive_tril(matrix)
    assert jnp.allclose(matrix, result)


def test_update_sqrt(iwp):
    """Test the square-root updates."""
    # Use sqrt(Q) as a dummy for a sqrt(C)
    A, SC = iwp.preconditioned_discretize_1d
    d = SC.shape[0]

    # Check square and non-square!
    for H in [A, A[:1]]:
        SC_new, kalman_gain, innov_chol = tornado.sqrt.update_sqrt(H, SC)
        assert isinstance(SC_new, jnp.ndarray)
        assert isinstance(kalman_gain, jnp.ndarray)
        assert isinstance(innov_chol, jnp.ndarray)
        assert SC_new.shape == (d, d)
        assert kalman_gain.shape == (d, H.shape[0])
        assert innov_chol.shape == (H.shape[0], H.shape[0])

        # expected:
        S = H @ SC @ SC.T @ H.T
        K = SC @ SC.T @ H.T @ jnp.linalg.inv(S)
        C = SC @ SC.T - K @ S @ K.T

        # Test SC
        assert jnp.allclose(SC_new @ SC_new.T, C)
        assert jnp.allclose(SC_new, jnp.tril(SC_new))
        assert jnp.all(jnp.diag(SC_new) >= 0)

        # Test K
        assert jnp.allclose(K, kalman_gain)

        # Test S
        assert jnp.allclose(innov_chol @ innov_chol.T, S)
        assert jnp.allclose(innov_chol, jnp.tril(innov_chol))
        assert jnp.all(jnp.diag(innov_chol) >= 0)


def test_batched_update_sqrt(iwp):

    H, process_noise_cholesky = iwp.preconditioned_discretize_1d
    d = process_noise_cholesky.shape[0]
    for transition_matrix in [H, H[:1]]:
        A = tornado.linops.BlockDiagonal(jnp.stack([transition_matrix] * 3))
        some_chol = tornado.linops.BlockDiagonal(
            jnp.stack([process_noise_cholesky.copy()] * 3)
        )

        chol, K, S = tornado.sqrt.batched_update_sqrt(
            A.array_stack,
            some_chol.array_stack,
        )
        print(chol.shape, K.shape, S.shape)
        assert isinstance(chol, jnp.ndarray)
        assert isinstance(K, jnp.ndarray)
        assert isinstance(S, jnp.ndarray)
        assert K.shape == (3, d, transition_matrix.shape[0])
        assert chol.shape == (3, d, d)
        assert S.shape == (3, transition_matrix.shape[0], transition_matrix.shape[0])

        chol_as_bd = tornado.linops.BlockDiagonal(chol)
        K_as_bd = tornado.linops.BlockDiagonal(K)
        S_as_bd = tornado.linops.BlockDiagonal(S)
        ref_chol, ref_K, ref_S = tornado.sqrt.update_sqrt(
            A.todense(), some_chol.todense()
        )

        assert jnp.allclose(K_as_bd.todense(), ref_K)

        # The Cholesky-factor of positive semi-definite matrices is only unique
        # up to column operations (e.g. column reordering), i.e. there could be slightly
        # different Cholesky factors in batched and non-batched versions.
        # Therefore, we only check that the results are valid Cholesky factors themselves
        assert jnp.allclose((S_as_bd @ S_as_bd.T).todense(), ref_S @ ref_S.T)
        assert jnp.all(jnp.diag(S_as_bd.todense()) >= 0.0)
        assert jnp.allclose(
            (chol_as_bd @ chol_as_bd.T).todense(), ref_chol @ ref_chol.T
        )
        assert jnp.all(jnp.diag(chol_as_bd.todense()) >= 0.0)
