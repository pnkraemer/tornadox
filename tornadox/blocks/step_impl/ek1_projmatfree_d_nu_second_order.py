"""EK1 for second-order equations."""

from functools import partial

import jax
import jax.numpy as jnp

from tornadox.blocks import sqrtutil


@partial(jax.jit, static_argnames=("f", "df", "num_derivatives"))
def attempt_step_forward_only(
    *, f, df, m, c_sqrtm, p, p_inv, a, q_sqrtm, num_derivatives
):
    """Attempt a full step.

    Extrapolate-then-correct.
    Includes error estimation.
    Includes a scalar, time-varying diffusion.

    Assumes a dimension-derivative ordering.
    """

    # Apply the pre-conditioner
    m, c_sqrtm = p_inv * m, p_inv[:, None] * c_sqrtm

    # Extrapolate mean and compute error
    m_ext = p * (a @ m)

    # Linearise ODE and estimate error
    fx, (Jx, Jy), m_obs = _linearise_ode_projmatfree(
        f=f, df=df, m_ext=m_ext, num_derivatives=num_derivatives
    )
    err, diff_sqrtm = _estimate_error_projmatfree(
        Jx=Jx,
        Jy=Jy,
        m_res=m_obs,
        q_sqrtm=p[:, None] * q_sqrtm,
        num_derivatives=num_derivatives,
    )

    # Extrapolate covariance
    c_sqrtm_ext = sqrtutil.sum_of_sqrtm_factors(S1=a @ c_sqrtm, S2=diff_sqrtm * q_sqrtm)

    # Un-apply the pre-conditioner
    c_sqrtm_ext = p[:, None] * c_sqrtm_ext

    # The final correction:
    h_mul_c_sqrtm = (
        c_sqrtm_ext[2 :: (num_derivatives + 1), :]
        - Jx @ c_sqrtm_ext[0 :: (num_derivatives + 1)]
        - Jy @ c_sqrtm_ext[1 :: (num_derivatives + 1)]
    )
    c_sqrtm_obs, (c_sqrtm_cor, g) = sqrtutil.correct_noisefree_matfree(
        c_sqrtm=c_sqrtm_ext, h_mul_c_sqrtm=h_mul_c_sqrtm
    )
    m_cor = m_ext - g @ m_obs

    return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err


@partial(jax.jit, static_argnames=("f", "df", "num_derivatives"))
def attempt_step(*, f, df, m, c_sqrtm, p, p_inv, a, q_sqrtm, num_derivatives):
    """Attempt a full step.

    Extrapolate-then-correct.
    Includes error estimation.
    Includes the backward transition, but no dense output.
    Includes a scalar, time-varying diffusion.
    """

    # Apply the pre-conditioner
    m, c_sqrtm = p_inv * m, p_inv[:, None] * c_sqrtm

    # Extrapolate mean and compute error
    m_ext = a @ m
    fx, (Jx, Jy), m_obs = _linearise_ode_projmatfree(
        f=f, df=df, m_ext=p * m_ext, num_derivatives=num_derivatives
    )
    err, diff_sqrtm = _estimate_error_projmatfree(
        Jx=Jx,
        Jy=Jy,
        m_res=m_obs,
        q_sqrtm=p[:, None] * q_sqrtm,
        num_derivatives=num_derivatives,
    )

    # Extrapolate covariance
    c_sqrtm_ext, (c_sqrtm_bw, g_bw) = sqrtutil.correct_noisy(
        c_sqrtm=c_sqrtm, h=a, r_sqrtm=diff_sqrtm * q_sqrtm
    )
    m_bw = m - g_bw @ m_ext
    backward_model = (g_bw, (m_bw, c_sqrtm_bw))

    # Un-apply the pre-conditioner
    m_ext, c_sqrtm_ext = p * m_ext, p[:, None] * c_sqrtm_ext

    # The final correction:
    h_mul_c_sqrtm = (
        c_sqrtm_ext[2 :: (num_derivatives + 1), :]
        - Jx @ c_sqrtm_ext[0 :: (num_derivatives + 1)]
        - Jy @ c_sqrtm_ext[1 :: (num_derivatives + 1)]
    )
    c_sqrtm_obs, (c_sqrtm_cor, g) = sqrtutil.correct_noisefree_matfree(
        c_sqrtm=c_sqrtm_ext, h_mul_c_sqrtm=h_mul_c_sqrtm
    )
    m_cor = m_ext - g @ m_obs

    return (
        (m_cor, c_sqrtm_cor),
        (m_obs, c_sqrtm_obs),
        (m_ext, c_sqrtm_ext),
        backward_model,
        err,
    )


@partial(jax.jit, static_argnames=("f", "df", "num_derivatives"))
def _linearise_ode_projmatfree(*, f, df, m_ext, num_derivatives):
    m_at_x = m_ext[0 :: (num_derivatives + 1)]
    m_at_y = m_ext[1 :: (num_derivatives + 1)]

    fx = f(m_at_x, m_at_y)
    Jx, Jy = df(m_at_x, m_at_y)
    m_obs = m_ext[2 :: (num_derivatives + 1)] - fx

    return fx, (Jx, Jy), m_obs


@partial(jax.jit, static_argnames=("num_derivatives",))
def _estimate_error_projmatfree(*, Jx, Jy, m_res, q_sqrtm, num_derivatives):
    s_sqrtm = (
        q_sqrtm[2 :: (num_derivatives + 1), :]
        - Jx @ q_sqrtm[0 :: (num_derivatives + 1), :]
        - Jy @ q_sqrtm[1 :: (num_derivatives + 1), :]
    )
    s_chol = sqrtutil.sqrtm_to_cholesky(St=s_sqrtm.T)

    res_white = jax.scipy.linalg.solve_triangular(s_chol.T, m_res, lower=False)
    diff = res_white.T @ res_white / res_white.size
    diff_sqrtm = jnp.sqrt(diff)

    error_estimate = diff_sqrtm * jnp.sqrt(jnp.einsum("jk,jk->j", s_chol, s_chol))
    return error_estimate, diff_sqrtm
