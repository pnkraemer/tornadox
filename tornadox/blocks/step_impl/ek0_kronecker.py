"""Kronecker EK0 implementation."""

from functools import partial

import jax
import jax.numpy as jnp

from tornadox.blocks import sqrtutil


@partial(jax.jit, static_argnames=("f",))
def attempt_step(*, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
    """A step with the 'KroneckerEK0'.

    Includes error estimation.
    Includes time-varying, scalar diffusion.
    """
    # m is an (nu+1,d) array. c_sqrtm is a (nu+1,nu+1) array.

    # Apply the pre-conditioner
    m, c_sqrtm = p_inv[:, None] * m, p_inv[:, None] * c_sqrtm

    # Predict the mean.
    # Immediately undo the preconditioning,
    # because it's served its purpose for the mean.
    # (It is not really necessary for the mean, to be honest.)
    m_ext_p = a @ m
    m_ext = p[:, None] * m_ext_p

    # Compute the error estimate
    m_obs = m_ext[1, :] - f(m_ext[0, :])
    err, diff_sqrtm = _estimate_error(m_res=m_obs, q_sqrtm=p[:, None] * q_sqrtm)

    # The full extrapolation:
    c_sqrtm_ext, (c_sqrtm_bw, g_bw) = sqrtutil.correct_noisy(
        c_sqrtm=c_sqrtm, h=a, r_sqrtm=diff_sqrtm * q_sqrtm
    )
    m_bw = m - g_bw @ m_ext_p  # note: uses the preconditioned extrapolation

    # Un-apply the pre-conditioner
    c_sqrtm_ext = p[:, None] * c_sqrtm_ext
    m_bw, c_sqrtm_bw = p[:, None] * m_bw, p[:, None] * c_sqrtm_bw
    g_bw = p[:, None] * g_bw * p_inv[None, :]
    backward_model = (g_bw, (m_bw, c_sqrtm_bw))

    # The final correction
    c_sqrtm_obs, (m_cor, c_sqrtm_cor) = _final_correction(
        m_obs=m_obs, m_ext=m_ext, c_sqrtm_ext=c_sqrtm_ext
    )

    return (
        (m_cor, c_sqrtm_cor),
        (m_obs, c_sqrtm_obs),
        (m_ext, c_sqrtm_ext),
        backward_model,
        err,
    )


@partial(jax.jit, static_argnames=("f",))
def attempt_step_forward_only(*, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
    """A step with the 'KroneckerEK0'.

    Includes error estimation.
    Includes time-varying, scalar diffusion.
    """
    # m is an (nu+1,d) array. c_sqrtm is a (nu+1,nu+1) array.

    # Apply the pre-conditioner
    m, c_sqrtm = p_inv[:, None] * m, p_inv[:, None] * c_sqrtm

    # Predict the mean.
    # Immediately undo the preconditioning,
    # because it's served its purpose for the mean.
    # (It is not really necessary for the mean, to be honest.)
    m_ext = p[:, None] * (a @ m)

    # Compute the error estimate
    m_obs = m_ext[1, :] - f(m_ext[0, :])
    err, diff_sqrtm = _estimate_error(m_res=m_obs, q_sqrtm=p[:, None] * q_sqrtm)

    # The full extrapolation:
    c_sqrtm_ext = sqrtutil.sum_of_sqrtm_factors(S1=a @ c_sqrtm, S2=diff_sqrtm * q_sqrtm)

    # Un-apply the pre-conditioner.
    # Now it is also done serving its purpose for the covariance.
    c_sqrtm_ext = p[:, None] * c_sqrtm_ext

    # The final correction
    c_sqrtm_obs, (m_cor, c_sqrtm_cor) = _final_correction(
        m_obs=m_obs, m_ext=m_ext, c_sqrtm_ext=c_sqrtm_ext
    )

    return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err


@jax.jit
def _final_correction(*, m_obs, m_ext, c_sqrtm_ext):
    # no fancy QR/sqrtm-stuff, because
    # the observation matrices have shape (): they are scalars.
    # The correction is almost free.
    s_sqrtm = c_sqrtm_ext[1, :]  # shape (n,)
    s = s_sqrtm @ s_sqrtm.T

    g = (c_sqrtm_ext @ s_sqrtm.T) / s  # shape (n,)
    c_sqrtm_cor = c_sqrtm_ext - g[:, None] * s_sqrtm[None, :]
    m_cor = m_ext - g[:, None] * m_obs[None, :]

    c_sqrtm_obs = jnp.sqrt(s)
    return c_sqrtm_obs, (m_cor, c_sqrtm_cor)


@jax.jit
def _estimate_error(*, m_res, q_sqrtm):
    s_sqrtm = q_sqrtm[1, :]
    s = s_sqrtm @ s_sqrtm.T
    diff = m_res.T @ m_res / (m_res.size * s)
    diff_sqrtm = jnp.sqrt(diff)
    error_estimate = diff_sqrtm * jnp.sqrt(s)
    return error_estimate, diff_sqrtm
