"""Gaussian filtering and smoothing routines."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg

from tornadox import sqrt


# Most popular for testing the smoother (and debugging other filters)
@jax.jit
def filter_step(m, sc, phi, sq, h, b, data):

    # Prediction
    m_pred = phi @ m
    x1 = phi @ sc
    sc_pred = sqrt.propagate_cholesky_factor(x1, sq)

    # Smoothing gain
    cross = (x1 @ sc.T).T
    sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

    # Update
    sc, kgain, _ = sqrt.update_sqrt(h, sc_pred)
    z = h @ m_pred + b
    m = m_pred - kgain @ (z - data)

    return m, sc, sgain, m_pred, sc_pred, x1


# Most popular for testing the square-root implementation
@jax.jit
def smoother_step_traditional(m, sc, m_fut, sc_fut, sgain, mp, scp):

    # Assemble full covariances
    c = sc @ sc.T
    c_fut = sc_fut @ sc_fut.T
    cp = scp @ scp.T

    # Update mean and cov
    new_mean = m + sgain @ (m_fut - mp)
    new_cov = c + sgain @ (c_fut - cp) @ sgain.T
    new_sc = jnp.linalg.cholesky(new_cov)

    return new_mean, new_sc


@jax.jit
def smoother_step_sqrt(m, sc, m_fut, sc_fut, sgain, sq, mp, x):

    # Update mean straightaway
    new_mean = m - sgain @ (mp - m_fut)

    # Compute covariance update with a QR decomposition
    d = m.shape[0]
    zeros = jnp.zeros((d, d))
    M = jnp.block(
        [
            [x.T, sc.T],
            [sq.T, zeros.T],
            [zeros.T, sc_fut.T @ sgain.T],
        ]
    )
    R = jax.scipy.linalg.qr(M, mode="r", pivoting=False)
    new_cov_cholesky = R[d : 2 * d, d:].T
    return new_mean, new_cov_cholesky
