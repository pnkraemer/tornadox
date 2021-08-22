"""Gaussian filtering and smoothing routines."""

import jax.numpy as jnp
import jax.scipy.linalg

from tornado import sqrt


def filter_step(m, sc, phi, sq, h, b, data):

    # Prediction
    m_pred = phi @ m
    x1 = phi @ sc
    sc_pred = sqrt.propagate_cholesky_factor(phi @ sc, sq)

    # Smoothing gain
    cross = x1 @ sc.T
    sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

    # Update
    sc, kgain, _ = sqrt.update_sqrt(h, sc_pred)
    z = h @ m_pred + b
    m = m_pred - kgain @ (z - data)

    return m, sc, sgain, m_pred, sc_pred


def smoother_step(m, sc, m_fut, sc_fut, sgain, mp, scp):

    c = sc @ sc.T
    c_fut = sc_fut @ sc_fut.T
    cp = scp @ scp.T

    new_mean = m - sgain @ (mp - m_fut)

    new_cov = c - sgain @ (cp - c_fut) @ sgain.T
    return new_mean, jnp.linalg.cholesky(new_cov)