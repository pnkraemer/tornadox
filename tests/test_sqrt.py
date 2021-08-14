"""Tests for square-root utilities."""

import jax.numpy as jnp
import pytest

import tornado


@pytest.fixture
def iwp():
    return tornado.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1, num_derivatives=2
    )


def test_propagate_cholesky_factor(iwp):
    transition_matrix, process_noise_cholesky = iwp.preconditioned_discretize_1d

    # dummy cholesky factor
    some_chol1 = process_noise_cholesky.copy()
    some_chol2 = process_noise_cholesky.copy()

    # First test: Non-optional S2
    chol = tornado.sqrt.propagate_cholesky_factor(
        S1=(transition_matrix @ some_chol1), S2=process_noise_cholesky
    )
    cov = (
        transition_matrix @ some_chol1 @ some_chol1.T @ transition_matrix.T
        + process_noise_cholesky @ process_noise_cholesky.T
    )
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.linalg.cholesky(cov), chol)
    assert jnp.all(jnp.diag(chol) > 0)

    # Second test: Optional S2
    chol = tornado.sqrt.propagate_cholesky_factor(S1=(transition_matrix @ some_chol2))
    cov = transition_matrix @ some_chol2 @ some_chol2.T @ transition_matrix.T

    # Relax tolerance because ill-conditioned...
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.linalg.cholesky(cov), chol, rtol=1e-4, atol=1e-4)
    assert jnp.all(jnp.diag(chol) > 0)


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

    # Check square and non-square!
    for H in [A, A[:1]]:
        SC_new, kalman_gain, innov_chol = tornado.sqrt.update_sqrt(H, SC)

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
