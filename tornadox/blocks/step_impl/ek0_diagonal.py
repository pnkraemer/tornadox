"""EK0 with independent dimensions."""

from functools import partial

import jax
import jax.numpy as jnp

from tornadox.blocks import sqrtutil


@partial(jax.jit, static_argnames=("f",))
def attempt_step_forward_only(*, f, m, c_sqrtm, p, p_inv, a, q_sqrtm):
    """A step with the 'DiagonalEK0'.

    Includes error estimation.
    Includes time-varying, multivariate diffusion.
    """
    # todo: move the below to a nice docstring
    # m is an (nu+1,d) array. c_sqrtm is a (d, nu+1,nu+1) array.
    #
    # The system matrices remain isotropic:
    #
    # a is an (nu+1, nu+1) array
    # q_sqrtm is an (nu+1, nu+1) array
    #
    # This is not necessarily a restriction, we just haven't needed
    # "batched" system matrices yet. If you do, make some noise!

    # Apply the pre-conditioner
    m, c_sqrtm = p_inv[:, None] * m, p_inv[None, :, None] * c_sqrtm

    # Extrapolate the mean.
    # Immediately undo the preconditioning,
    # because it's served its purpose for the mean.
    # (It is not really necessary for the mean, to be honest.)
    m_ext = p[:, None] * (a @ m)

    # Compute the error estimate
    m_obs = m_ext[1, :] - f(m_ext[0, :])
    err, diff_sqrtm = _estimate_error(
        m_res=m_obs, q_sqrtm=p[None, :, None] * q_sqrtm[None, ...]
    )

    # Extrapolate the covariance:
    #
    # Start by broadcasting the vector-valued diffusion into q_sqrtm
    q_sqrtm_diff = diff_sqrtm[:, None, None] * q_sqrtm[None, ...]
    #
    # And proceed by by computing a @ c for all batch dimensions.
    ac_sqrtm = a[None, ...] @ c_sqrtm
    c_sqrtm_ext = sqrtutil.sum_of_sqrtm_factors_batched(ac_sqrtm, q_sqrtm_diff)

    # Un-apply the pre-conditioner.
    # Now it is also done serving its purpose for the covariance.
    c_sqrtm_ext = p[None, :, None] * c_sqrtm_ext

    # The final correction.
    # Note how we slice the 1st dimension out of c_sqrtm_ext,
    # but how we "add" a fake-dimension back in. This is because
    # tools in sqrtutil.py expect ndim=2 arrays.
    # The fake-dimension will be removed straightaway.
    c_sqrtm_obs, (c_sqrtm_cor, g) = sqrtutil.correct_noisefree_matfree_batched(
        c_sqrtm_ext[:, [1], :], c_sqrtm_ext
    )
    c_sqrtm_obs = c_sqrtm_obs[:, 0, 0]  # from shape (d,1,1) to shape (d,)

    # Finally, correct the mean. This involves nasty broadcasting,
    # so see wrote a thorough explanation and moved it into the function below.
    m_cor = _correct_mean(m=m_ext, g=g, m_res=m_obs)
    return (m_cor, c_sqrtm_cor), (m_obs, c_sqrtm_obs), err


@jax.jit
def _correct_mean(*, m, g, m_res):
    # m is an (n,d) array
    # g is an (d, n, 1) array
    # m_res is an (d,) array
    #
    # The broadcasting in here is nasty.
    #
    # So what is going on? Let's see:

    # The correction term has shape (d,n,1),
    # because (d, n, 1) x (d, 1, 1).
    # It means that each dimension is treated independently,
    # and for each dimension there is a (n, 1) gain, and (1,) observations.
    correction = g @ m_res[:, None, None]

    # The "1" extra dimension has served its purpose.
    # We may just remove it.
    # The below has shape (d, n)
    # [because formerly, it had shape (d, n, 1)].
    correction = correction[:, :, 0]

    # At last, we transpose the correction term,
    # because the mean is always stored in (n, d) shape,
    # but the computations above naturally had (d, n) shape.
    # (This made more sense for broadcasting reasons.)
    new_mean = m - correction.T
    return new_mean


@jax.jit
def _estimate_error(*, m_res, q_sqrtm):
    s_sqrtm = q_sqrtm[0, 1, :]  # shape (n,)
    s = s_sqrtm @ s_sqrtm.T  # shape ()
    s_chol = jnp.sqrt(s)  # shape ()
    diff = (m_res / s_chol) ** 2 / m_res.size  # shape (d,)
    diff_sqrtm = jnp.sqrt(diff)  # shape (d,)
    error_estimate = diff_sqrtm * s_chol[None, ...]  # shape (d,)
    return error_estimate, diff_sqrtm
