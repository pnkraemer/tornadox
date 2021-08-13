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
