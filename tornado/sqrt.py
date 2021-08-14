"""Square-root transition utility functions."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def propagate_batched_cholesky_factor(batched_S1, batched_S2=None):
    """Propagate Cholesky factors for batches of matrix-square-roots."""
    if batched_S2 is None:
        return jnp.stack([propagate_cholesky_factor(s1) for s1 in batched_S1])
    return jnp.stack(
        [propagate_cholesky_factor(s1, s2) for s1, s2 in zip(batched_S1, batched_S2)]
    )


def propagate_cholesky_factor(S1, S2=None):
    """Compute Cholesky factor of A @ SC @ SC.T @ A.T + SQ @ SQ.T"""
    if S2 is not None:
        stacked_up = jnp.vstack((S1.T, S2.T))
    else:
        stacked_up = jnp.vstack(S1.T)
    upper_sqrtm = jnp.linalg.qr(stacked_up, mode="r")
    lower_sqrtm = upper_sqrtm.T
    return tril_to_positive_tril(lower_sqrtm)


def tril_to_positive_tril(tril_mat):
    r"""Orthogonally transform a lower-triangular matrix into a lower-triangular matrix with positive diagonal.
    In other words, make it a valid lower Cholesky factor.
    The name of the function is based on `np.tril`.
    """
    diag = jnp.diag(tril_mat)
    d = jnp.sign(diag)

    # Like numpy, JAX assigns sign 0 to 0.0, which eliminate entire rows in the operation below.
    d = jax.ops.index_add(d, d == 0, 1.0)

    # Fast(er) multiplication with a diagonal matrix from the right via broadcasting.
    with_pos_diag = tril_mat * d[None, :]
    return with_pos_diag


def update_sqrt(transition_matrix, cov_cholesky):
    """Compute the update step with noise-free linear observation models in square-root form.

    Parameters
    ----------
    transition_matrix
        Transition matrix. Shape (d_out, d_in)
    cov_cholesky
        Cholesky factor of the current (usually, the predicted) covariance. Shape (d_in, d_in)

    Returns
    -------
    jnp.ndarray
        Cholesky factor of the posterior covariance. Shape (d_out, d_out).
    jnp.ndarray
        Kalman gain. Shape (d_in, d_out).
    jnp.ndarray
        Cholesky factor of the innovation covariance matrix. Shape (d_out, d_out).
    """
    output_dim, input_dim = transition_matrix.shape
    zeros_bottomleft = jnp.zeros((output_dim, input_dim))
    zeros_bottomright = jnp.zeros((input_dim, input_dim))

    blockmat = jnp.block(
        [
            [cov_cholesky.T @ transition_matrix.T, cov_cholesky.T],
            [zeros_bottomleft.T, zeros_bottomright.T],
        ]
    )
    big_triu = jax.scipy.linalg.qr(blockmat, mode="r", pivoting=False)
    R3 = big_triu[
        output_dim : (output_dim + input_dim), output_dim : (output_dim + input_dim)
    ]
    R1 = big_triu[:output_dim, :output_dim]
    R2 = big_triu[:output_dim, output_dim:]
    gain = jax.scipy.linalg.solve_triangular(R1, R2, lower=False).T
    return tril_to_positive_tril(R3.T), gain, tril_to_positive_tril(R1.T)
