"""Tests for square-root utilities."""

import jax
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
    some_chol = process_noise_cholesky.copy()
    chol = tornado.sqrt.propagate_cholesky_factor(
        A=transition_matrix, SC=some_chol, SQ=process_noise_cholesky
    )
    cov = (
        transition_matrix @ some_chol @ some_chol.T @ transition_matrix.T
        + process_noise_cholesky @ process_noise_cholesky.T
    )

    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.linalg.cholesky(cov), chol)


def test_tril_to_positive_tril():

    matrix = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
        ]
    )
    result = tornado.sqrt.tril_to_positive_tril(matrix)
    assert jnp.allclose(matrix, result)
