"""Utility functions for square-root implementations."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp


@jax.jit
def correct_noisefree(*, h, c_sqrtm):
    return correct_noisefree_matfree(h_mul_c_sqrtm=h @ c_sqrtm, c_sqrtm=c_sqrtm)


@jax.jit
def correct_noisefree_matfree(*, h_mul_c_sqrtm, c_sqrtm):
    c_sqrtm_obs = sqrtm_to_cholesky(St=h_mul_c_sqrtm.T)
    g = jsp.linalg.cho_solve((c_sqrtm_obs, True), h_mul_c_sqrtm @ c_sqrtm.T).T
    c_sqrtm_cor = sqrtm_to_cholesky(St=(c_sqrtm - g @ h_mul_c_sqrtm).T)
    return c_sqrtm_obs, (c_sqrtm_cor, g)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0), out_axes=(0, (0, 0)))
def correct_noisefree_matfree_batched(h_mul_c_sqrtm, c_sqrtm):
    """Same as correct_noisefree_matfree, but batched."""
    # careful: vmapped qr() (which this function calls)
    # is not parallelised on GPU. See
    # https://github.com/google/jax/issues/8542
    return correct_noisefree_matfree(h_mul_c_sqrtm=h_mul_c_sqrtm, c_sqrtm=c_sqrtm)


@jax.jit
def correct_noisy(*, h, c_sqrtm, r_sqrtm):
    return correct_noisy_matfree(
        h_matmul_c_sqrtm=h @ c_sqrtm, c_sqrtm=c_sqrtm, r_sqrtm=r_sqrtm
    )


@jax.jit
def correct_noisy_matfree(*, h_matmul_c_sqrtm, c_sqrtm, r_sqrtm):

    output_dim, input_dim = h_matmul_c_sqrtm.shape

    blockmat = jnp.block(
        [
            [r_sqrtm, h_matmul_c_sqrtm],
            [jnp.zeros((input_dim, output_dim)), c_sqrtm],
        ]
    ).T

    R = jnp.linalg.qr(blockmat, mode="r")

    R1 = R[:output_dim, :output_dim]  # observed RV
    R12 = R[:output_dim, output_dim:]  # something like the crosscov
    R3 = R[output_dim:, output_dim:]  # corrected RV

    # todo: what is going on here???
    #  why lstsq? The matrix should be well-conditioned.
    gain = jnp.linalg.lstsq(R1, R12)[0].T

    c_sqrtm_cor = _make_diagonal_positive(L=R3.T)
    c_sqrtm_obs = _make_diagonal_positive(L=R1.T)
    return c_sqrtm_obs, (c_sqrtm_cor, gain)


@jax.jit
def sum_of_sqrtm_factors(*, S1, S2):
    """Compute Cholesky factor of S1 @ S1.T + S2 @ S2.T"""
    stacked_up = jnp.vstack((S1.T, S2.T))
    return sqrtm_to_cholesky(St=stacked_up)


@jax.jit
@partial(jax.vmap, in_axes=(0, 0), out_axes=0)
def sum_of_sqrtm_factors_batched(S1, S2):
    """Batched version of sum_of_sqrtm_factors."""
    # careful: vmapped qr() (which this function calls)
    # is not parallelised on GPU. See
    # https://github.com/google/jax/issues/8542
    return sum_of_sqrtm_factors(S1=S1, S2=S2)


@jax.jit
def sqrtm_to_cholesky(*, St):
    """Assume that St=S^\top is a 'right' matrix-square-root.

    I.e. assume M = S S^\top.
    """
    upper_sqrtm = jnp.linalg.qr(St, mode="r")
    lower_sqrtm = upper_sqrtm.T
    return _make_diagonal_positive(L=lower_sqrtm)


@jax.jit
def _make_diagonal_positive(*, L):
    s = jnp.sign(jnp.diag(L))
    x = jnp.where(s == 0, 1, s)
    return L * x[None, ...]
