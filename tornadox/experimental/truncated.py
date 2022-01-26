from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg

from tornadox import ek1, iwp, odefilter, rv, sqrt
from tornadox.experimental import linops


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


class EarlyTruncationEK1(odefilter.ODEFilter):
    """Use full Jacobians for mean-updates, but truncate cleverly to enforce a block-diagonal posterior covariance.

    "Cleverly" means:
    Truncate the Jacobian into a diagonal matrix after the mean update
    and recompute posterior covariance which now has block-diagonal structure (see DiagonalEK1).
    (This also means that for the covariance update, we use the inverse of the diagonal of S, not the diagonal of the inverse of S.)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0_1d = None
        self.P1_1d = None
        self.P0 = None
        self.P1 = None

    def initialize(self, f, t0, tmax, y0, df, df_diagonal):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives, wiener_process_dimension=y0.shape[0]
        )
        self.P0_1d = self.iwp.projection_matrix_1d(0)
        self.P1_1d = self.iwp.projection_matrix_1d(1)

        d = self.iwp.wiener_process_dimension
        self.P0 = linops.BlockDiagonal(jnp.stack([self.P0_1d] * d))
        self.P1 = linops.BlockDiagonal(jnp.stack([self.P1_1d] * d))

        extended_dy0, cov_sqrtm = self.init(
            f=f,
            df=df,
            y0=y0,
            t0=t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_sqrtm = jnp.stack([cov_sqrtm] * d)
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odefilter.ODEFilterState(
            t=t0,
            y=new_rv,
            error_estimate=jnp.nan * jnp.ones(self.iwp.wiener_process_dimension),
            reference_state=jnp.nan * jnp.ones(self.iwp.wiener_process_dimension),
        )

    @partial(jax.jit, static_argnums=(0, 3, 7, 8))
    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):
        d = self.iwp.wiener_process_dimension
        n = self.iwp.num_derivatives + 1

        # Assemble preconditioner
        P_1d, Pinv_1d = self.iwp.nordsieck_preconditioner_1d(dt=dt)
        P = linops.BlockDiagonal(jnp.stack([P_1d] * d))
        Pinv = linops.BlockDiagonal(jnp.stack([Pinv_1d] * d))
        # assert isinstance(P, linops.BlockDiagonal)
        # assert isinstance(Pinv, linops.BlockDiagonal)
        # assert P.array_stack.shape == (d, n, n)
        # assert Pinv.array_stack.shape == (d, n, n)

        # Assemble projection-preconditioner combo
        P0_1d = self.P0_1d @ P_1d
        P1_1d = self.P1_1d @ P_1d
        P0 = linops.BlockDiagonal(jnp.stack([P0_1d] * d))
        P1 = linops.BlockDiagonal(jnp.stack([P1_1d] * d))
        # assert isinstance(P0, linops.BlockDiagonal)
        # assert isinstance(P1, linops.BlockDiagonal)
        # assert P0.array_stack.shape == (d, 1, n)
        # assert P1.array_stack.shape == (d, 1, n)

        # Extract system matrices
        A, SQ = self.iwp.preconditioned_discretize_1d
        A = linops.BlockDiagonal(jnp.stack([A] * d))
        SQ = linops.BlockDiagonal(jnp.stack([SQ] * d))
        # assert isinstance(A, linops.BlockDiagonal)
        # assert isinstance(SQ, linops.BlockDiagonal)
        # assert A.array_stack.shape == (d, n, n)
        # assert SQ.array_stack.shape == (d, n, n)

        # Extract previous states and pull them into "preconditioned space"
        # assert isinstance(state.y.cov_sqrtm, linops.BlockDiagonal)
        # assert state.y.cov_sqrtm.array_stack.shape == (d, n, n)
        m = Pinv @ state.y.mean.reshape((-1,), order="F")
        SC = Pinv @ linops.BlockDiagonal(state.y.cov_sqrtm)
        # assert isinstance(SC, linops.BlockDiagonal)
        # assert SC.array_stack.shape == (d, n, n)

        # Predict [mean]
        m_pred = A @ m

        # Evaluate ODE
        t = state.t + dt
        m_at = P0 @ m_pred
        f = f(t, m_at)
        Jx = df(t, m_at)  # Use full Jacobian here!

        # Evaluate H @ sth manually (i.e. pseudo-lazily),
        # because P0/P1 slice, and Jx @ sth is dense matmul.
        b = Jx @ m_at - f
        z = P1 @ m_pred - Jx @ (P0 @ m_pred) + b
        # assert isinstance(z, jnp.ndarray)
        # assert z.shape == (d,)

        # Calibrate: Extract P0 @ SC and P1 @ SC
        # only then densify and apply Jx @ sth.
        SQ0_dense = (P0 @ SQ).todense()
        SQ1_dense = (P1 @ SQ).todense()
        # assert SQ0_dense.shape == (d, d * n)
        # assert SQ1_dense.shape == (d, d * n)
        JxSQ0 = Jx @ SQ0_dense
        # assert JxSQ0.shape == (d, d * n)
        S_sqrtm = sqrt.sqrtm_to_cholesky((SQ1_dense - JxSQ0).T)
        # assert S_sqrtm.shape == (d, d)
        whitened_res = jax.scipy.linalg.solve_triangular(S_sqrtm.T, z, lower=False)
        # assert whitened_res.shape == (d,)
        sigma_squared = whitened_res.T @ whitened_res / d
        sigma = jnp.sqrt(sigma_squared)
        # assert sigma_squared.shape == ()
        # assert sigma.shape == ()
        # assert sigma_squared >= 0.0
        # assert sigma >= 0.0

        # Assemble full S for the error estimate,
        # because we need the diagonal elements of a dense matrix S
        S = S_sqrtm @ S_sqrtm.T
        error_estimate = sigma * jnp.sqrt(jnp.diag(S))
        # assert isinstance(error_estimate, jnp.ndarray)
        # assert error_estimate.shape == (d,)
        # assert jnp.all(error_estimate >= 0.0)

        # Predict [cov]
        batched_sc_pred = sqrt.batched_propagate_cholesky_factor(
            (A @ SC).array_stack, sigma * SQ.array_stack
        )
        SC_pred = linops.BlockDiagonal(batched_sc_pred)
        # assert isinstance(SC_pred, linops.BlockDiagonal)
        # assert SC_pred.array_stack.shape == (d, n, n)

        # Compute innovation matrix and Kalman gain
        # First project, then apply Jx (see above)
        SC_pred0_dense = (P0 @ SC_pred).todense()
        SC_pred1_dense = (P1 @ SC_pred).todense()
        # assert SC_pred0_dense.shape == (d, d * n)
        # assert SC_pred1_dense.shape == (d, d * n)
        JxSC_pred0 = Jx @ SC_pred0_dense
        # assert JxSC_pred0.shape == (d, d * n)
        S_sqrtm = sqrt.sqrtm_to_cholesky((SC_pred1_dense - JxSC_pred0).T)
        # assert S_sqrtm.shape == (d, d)

        # Dense cross-covariance; again, apply P0 and P1 separately from Jx
        Cminus = SC_pred @ SC_pred.T
        R0 = P0 @ Cminus
        R1 = P1 @ Cminus
        crosscov_transposed = R1.todense() - Jx @ R0.todense()
        crosscov = crosscov_transposed.T
        # assert crosscov.shape == (d * n, d), crosscov.shape

        # Mean update; contains the only solve() with a dense dxd matrix in the whole script
        # Maybe we turn this into a call to CG at some point
        # (it should be possible to use sparsity of Jx here; ping @nk for a discussion)
        solved = jax.scipy.linalg.cho_solve((S_sqrtm, True), z)
        new_mean = m_pred - crosscov @ solved
        # assert isinstance(new_mean, jnp.ndarray)
        # assert new_mean.shape == (d * n,)

        # Truncate the hell out of S and K
        # Extract the diagonal from Jx, and do the rest as in DiagonalEK1.attempt_step()
        # Replicate the respective parts from DiagonalEK1()
        Jx_as_diag = linops.BlockDiagonal(jnp.diag(Jx).reshape((-1, 1, 1)))
        H = P1 - Jx_as_diag @ P0
        S_as_diag = (H @ SC_pred) @ (H @ SC_pred).T
        crosscov = Cminus @ H.T
        kalman_gain = crosscov @ linops.BlockDiagonal(1.0 / S_as_diag.array_stack)
        # assert isinstance(kalman_gain, linops.BlockDiagonal)
        # assert kalman_gain.array_stack.shape == (d, n, 1)

        # Update covariance
        I = linops.BlockDiagonal(jnp.stack([jnp.eye(n, n)] * d))
        cov_sqrtm = (I - kalman_gain @ H) @ SC_pred
        # assert isinstance(cov_sqrtm, linops.BlockDiagonal)
        # assert cov_sqrtm.array_stack.shape == (d, n, n)

        # Push mean and covariance back into "normal space"
        new_mean = (P @ new_mean).reshape((n, d), order="F")
        cov_sqrtm = (P @ cov_sqrtm).array_stack
        # assert isinstance(cov_sqrtm, linops.BlockDiagonal)
        # assert cov_sqrtm.array_stack.shape == (d, n, n)

        y1 = jnp.abs(self.P0 @ state.y.mean.reshape((-1,), order="F"))
        y2 = jnp.abs(self.P0 @ new_mean.reshape((-1,), order="F"))
        reference_state = jnp.maximum(y1, y2)
        # assert isinstance(reference_state, jnp.ndarray)
        # assert reference_state.shape == (d,)
        # assert jnp.all(reference_state >= 0.0), reference_state

        # Return new state
        new_rv = rv.BatchedMultivariateNormal(new_mean, cov_sqrtm)

        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)

        new_state = odefilter.ODEFilterState(
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
        return new_state, info_dict
