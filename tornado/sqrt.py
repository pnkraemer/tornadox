"""Square-root transition utility functions."""

import jax
import jax.numpy as jnp
import jax.scipy.linalg


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
    big_triu = jnp.linalg.qr(blockmat, mode="r")
    R3 = big_triu[
        output_dim : (output_dim + input_dim), output_dim : (output_dim + input_dim)
    ]
    R1 = big_triu[:output_dim, :output_dim]
    R2 = big_triu[:output_dim, output_dim:]
    gain = jax.scipy.linalg.solve_triangular(R1, R2, lower=False).T
    return tril_to_positive_tril(R3.T), gain, tril_to_positive_tril(R1.T)


"""

        # Smoothing updates need the gain, but
        # filtering updates "compute their own".
        # Thus, if we are doing smoothing (|cov_obtained|>0) an the gain is not provided,
        # make an extra prediction to compute the gain.
        if gain is None:
            if np.linalg.norm(rv_obtained.cov) > 0:
                rv_forwarded, info_forwarded = self.forward_rv(
                    rv, t=t, compute_gain=True, _diffusion=_diffusion
                )
                gain = info_forwarded["gain"]
            else:
                gain = np.zeros((len(rv.mean), len(rv_obtained.mean)))

        state_trans = self.state_trans_mat_fun(t)
        proc_noise_chol = np.sqrt(_diffusion) * self.proc_noise_cov_cholesky_fun(t)
        shift = self.shift_vec_fun(t)

        chol_past = rv.cov_cholesky
        chol_obtained = rv_obtained.cov_cholesky

        output_dim = self.output_dim
        input_dim = self.input_dim

        zeros_bottomleft = np.zeros((output_dim, output_dim))
        zeros_middleright = np.zeros((output_dim, input_dim))

        blockmat = np.block(
            [
                [chol_past.T @ state_trans.T, chol_past.T],
                [proc_noise_chol.T, zeros_middleright],
                [zeros_bottomleft, chol_obtained.T @ gain.T],
            ]
        )
        big_triu = np.linalg.qr(blockmat, mode="r")
        new_chol_triu = big_triu[
            output_dim : (output_dim + input_dim), output_dim : (output_dim + input_dim)
        ]

        # If no initial gain was provided, compute it from the QR-results
        # This is required in the Kalman update, where, other than in the smoothing update,
        # no initial gain was provided.
        # Recall that above, gain was set to zero in this setting.
        if np.linalg.norm(gain) == 0.0:
            R1 = big_triu[:output_dim, :output_dim]
            R12 = big_triu[:output_dim, output_dim:]
            gain = scipy.linalg.solve_triangular(R1, R12, lower=False).T

        new_mean = rv.mean + gain @ (rv_obtained.mean - state_trans @ rv.mean - shift)
        new_cov_cholesky = tril_to_positive_tril(new_chol_triu.T)
        new_cov = new_cov_cholesky @ new_cov_cholesky.T

        info = {"rv_forwarded": rv_forwarded}
        return randvars.Normal(new_mean, new_cov, cov_cholesky=new_cov_cholesky), info

"""
