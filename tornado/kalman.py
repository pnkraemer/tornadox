"""Gaussian filtering and smoothing routines."""

import jax.scipy.linalg

from tornado import sqrt


def filter_step_1d(m_1d, sc_1d, phi_1d, sq_1d, h_1d, b_1d, data):

    # Prediction
    m_pred = phi_1d @ m_1d
    x1 = phi_1d @ sc_1d
    sc_pred = sqrt.propagate_cholesky_factor(phi_1d @ sc_1d, sq_1d)

    # Smoothing gain
    cross = x1 @ sc_1d.T
    sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

    # Update
    sc, kgain, _ = sqrt.update_sqrt(h_1d, sc_pred)
    z = h_1d @ m_pred + b_1d
    m = m_pred - kgain @ (z - data)

    return m, sc, sgain
