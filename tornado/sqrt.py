"""Square-root transition utility functions."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


def propagate_cholesky_factor(S1, S2):
    """Compute Cholesky factor of A @ SC @ SC.T @ A.T + SQ @ SQ.T"""
    stacked_up = jnp.vstack((S1.T, S2.T))
    return sqrtm_to_cholesky(stacked_up)


def sqrtm_to_cholesky(St):
    """Assume that St=S^\top is a 'right' matrix-square-root.

    I.e. assume M = S S^\top.
    """
    upper_sqrtm = jnp.linalg.qr(St, mode="r")
    lower_sqrtm = upper_sqrtm.T
    return tril_to_positive_tril(lower_sqrtm)


# Batch the propagation function with jax.vmap magic.
batched_propagate_cholesky_factor = jax.vmap(
    propagate_cholesky_factor, in_axes=(0, 0), out_axes=0
)
batched_sqrtm_to_cholesky = jax.vmap(sqrtm_to_cholesky, in_axes=0, out_axes=0)


def tril_to_positive_tril(tril_mat):
    r"""Orthogonally transform a lower-triangular matrix into a lower-triangular matrix with positive diagonal.
    In other words, make it a valid lower Cholesky factor.
    The name of the function is based on `np.tril`.
    """
    diag = jnp.diag(tril_mat)

    # Like numpy, JAX assigns sign 0 to 0.0, which would eliminate entire rows
    # in the operation below. By adding a machine precision epsilon, there is no 0.
    d = jnp.sign(diag + jnp.finfo(diag[0].dtype).eps)

    # Fast(er) multiplication with a diagonal matrix from the right via broadcasting.
    with_pos_diag = tril_mat * d[None, :]
    return with_pos_diag


@jax.jit
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


# Todo: replace with a jax.vmap somehow
# The difficulty are the multiple outputs.
# Without the innov chol, cov_chol and kgain could be stacked together, vmapped, and extracted cleverly
# With all three however, this does not work, because there is no consistent way of
# stacking shapes (d, d), (d, l), (l, l) into a single array?!
# Therefore, do this with a loop for now and let jax.jit do the magic if speed was desired.
def batched_update_sqrt(batched_transition_matrix, batched_cov_cholesky):
    cov_chol, kgain, innov_chol = [], [], []
    for (A, SC) in zip(batched_transition_matrix, batched_cov_cholesky):
        c, k, s = update_sqrt(A, SC)
        cov_chol.append(c)
        kgain.append(k)
        innov_chol.append(s)
    return jnp.stack(cov_chol), jnp.stack(kgain), jnp.stack(innov_chol)
