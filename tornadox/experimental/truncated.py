from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg

from tornadox import ek1, sqrt


class TruncationEK1(ek1.BatchedEK1):
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def attempt_unit_step(self, f, df, df_diagonal, p_1d_raw, m, sc, t):
        m_pred = self.predict_mean(m, phi_1d=self.phi_1d)
        f, Jx, z = self.evaluate_ode(t=t, f=f, df=df, p_1d_raw=p_1d_raw, m_pred=m_pred)
        error, sigma = self.estimate_error(
            p_1d_raw=p_1d_raw,
            Jx=Jx,
            sq_bd=self.batched_sq,
            z=z,
        )
        sc_pred = self.predict_cov_sqrtm(
            sc_bd=sc, phi_1d=self.phi_1d, sq_bd=sigma * self.batched_sq
        )
        ss, kgain = self.observe_cov_sqrtm(Jx=Jx, p_1d_raw=p_1d_raw, sc_bd=sc_pred)
        new_mean = self.correct_mean(m=m_pred, kgain=kgain, z=z)
        cov_sqrtm = self.correct_cov_sqrtm(
            Jx=Jx,
            p_1d_raw=p_1d_raw,
            sc_bd=sc_pred,
            kgain=kgain,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
        return new_mean, cov_sqrtm, error, info_dict

    # Low level implementations

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df, p_1d_raw, m_pred):
        m_pred_no_precon = p_1d_raw[:, None] * m_pred
        m_at = m_pred_no_precon[0]
        fx = f(t, m_at)
        z = m_pred_no_precon[1] - fx
        Jx = df(t, m_at)
        return fx, Jx, z

    @staticmethod
    @jax.jit
    def estimate_error(p_1d_raw, Jx, sq_bd, z):

        sq_bd_no_precon = p_1d_raw[None, :, None] * sq_bd  # shape (d,n,n)
        q = sq_bd_no_precon @ jnp.transpose(sq_bd_no_precon, axes=(0, 2, 1))
        q_00 = q[:, 0, 0]
        q_01 = q[:, 0, 1]
        q_11 = q[:, 1, 1]
        s = (
            jnp.diag(q_11)
            - Jx * q_01[None, :]
            - q_01[:, None] * Jx.T
            + (Jx * q_00[None, :]) @ Jx.T
        )

        # Careful!! Here is one of the expensive bits!
        s_sqrtm = jax.scipy.linalg.cholesky(s, lower=True)
        xi = jax.scipy.linalg.solve_triangular(s_sqrtm.T, z, lower=False)

        sigma_squared = xi.T @ xi / xi.shape[0]  # shape ()
        sigma = jnp.sqrt(sigma_squared)  # shape ()
        error_estimate = sigma * jnp.sqrt(jnp.diag(s))  # shape (d,)

        return error_estimate, sigma

    @staticmethod
    @jax.jit
    def observe_cov_sqrtm(p_1d_raw, Jx, sc_bd):

        # Assemble S = H C- H.T efficiently
        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        c = sc_bd_no_precon @ jnp.transpose(sc_bd_no_precon, axes=(0, 2, 1))
        c_00 = c[:, 0, 0]
        c_01 = c[:, 0, 1]
        c_10 = c[:, 1, 0]
        c_11 = c[:, 1, 1]
        s = (
            jnp.diag(c_11)
            - Jx * c_01[None, :]
            - c_10[:, None] * Jx.T
            + (Jx * c_00[None, :]) @ Jx.T
        )

        # Assemble C- H.T = \sqrt(C-) \sqrt(C-).T H.T efficiently
        # Careful!! Here is one of the expensive bits!
        c_p = sc_bd @ jnp.transpose(sc_bd_no_precon, axes=(0, 2, 1))
        c_p_0 = jnp.transpose(c_p, axes=(0, 2, 1))[:, 0, :]
        c_p_1 = jnp.transpose(c_p, axes=(0, 2, 1))[:, 1, :]

        c_p_0_dense = jax.scipy.linalg.block_diag(*c_p_0[:, None, :])
        c_p_1_dense = jax.scipy.linalg.block_diag(*c_p_1[:, None, :])
        cross = (c_p_1_dense - Jx @ c_p_0_dense).T
        s_sqrtm = jax.scipy.linalg.cholesky(s, lower=True)
        kgain = jax.scipy.linalg.cho_solve((s_sqrtm.T, False), cross.T).T
        return s_sqrtm, kgain

    @staticmethod
    @jax.jit
    def correct_mean(m, kgain, z):
        correction = kgain @ z  # shape (d*n, d)
        new_mean = m - correction.reshape(m.shape, order="F")  # shape (n,d)
        return new_mean

    @staticmethod
    @jax.jit
    def correct_cov_sqrtm(p_1d_raw, Jx, sc_bd, kgain):

        # Evaluate H P \sqrt(C)
        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        sc0 = jax.scipy.linalg.block_diag(*sc_bd_no_precon_0[:, None, :])
        sc1 = jax.scipy.linalg.block_diag(*sc_bd_no_precon_1[:, None, :])
        Jx_sc0 = Jx @ sc0
        h_sc = sc1 - Jx_sc0

        # Evaluate \sqrt(C) - K (H P \sqrt(C))
        sc_dense = jax.scipy.linalg.block_diag(*sc_bd)
        new_cov_sqrtm = sc_dense - kgain @ h_sc

        # Split into d rows and QR-decompose the rows into valid blocks
        d = Jx.shape[0]
        split_covs = jnp.stack(jnp.split(new_cov_sqrtm, d, axis=0))
        new_sc = sqrt.batched_sqrtm_to_cholesky(
            jnp.transpose(split_covs, axes=(0, 2, 1))
        )
        return new_sc
