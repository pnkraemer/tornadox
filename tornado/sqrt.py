"""Square-root transition utility functions."""

import jax
import jax.numpy as jnp


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
