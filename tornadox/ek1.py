"""EK1 solvers."""

from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg

from tornadox import init, iwp, linops, odefilter, rv, sqrt


class ReferenceEK1(odefilter.ODEFilter):
    """Naive, reference EK1 implementation. Use this to test against."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.P1 = None

    def initialize(self, ivp):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.P0 = self.iwp.projection_matrix(0)
        self.P1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0  # .reshape((-1,), order="F")
        y = rv.MultivariateNormal(mean, jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm))
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        # Extract system matrices
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, SQ = self.iwp.preconditioned_discretize
        t = state.t + dt
        n, d = self.num_derivatives + 1, state.ivp.dimension

        # Pull states into preconditioned state
        m, SC = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        cov_cholesky, error_estimate, new_mean = self.attempt_unit_step(
            A, P, SC, SQ, m, state, t
        )

        # Push back to non-preconditioned state
        cov_cholesky = P @ cov_cholesky
        new_mean = P @ new_mean
        new_mean = new_mean.reshape((n, d), order="F")
        new_rv = rv.MultivariateNormal(new_mean, cov_cholesky)

        y1 = jnp.abs(state.y.mean[0])
        y2 = jnp.abs(new_mean[0])
        reference_state = jnp.maximum(y1, y2)

        # Return new state
        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
        return new_state, info_dict

    def attempt_unit_step(self, A, P, SC, SQ, m, state, t):
        m_pred = self.predict_mean(m=m, phi=A)
        H, z = self.evaluate_ode(
            t=t,
            f=state.ivp.f,
            df=state.ivp.df,
            p=P,
            m_pred=m_pred,
            e0=self.P0,
            e1=self.P1,
        )
        error_estimate, sigma = self.estimate_error(h=H, sq=SQ, z=z)
        SC_pred = self.predict_cov_sqrtm(sc=SC, phi=A, sq=sigma * SQ)
        cov_cholesky, Kgain, sqrt_S = sqrt.update_sqrt(H, SC_pred)
        new_mean = m_pred - Kgain @ z
        return cov_cholesky, error_estimate, new_mean

    # Low level functions

    @staticmethod
    @jax.jit
    def predict_mean(m, phi):
        return phi @ m

    @staticmethod
    @jax.jit
    def predict_cov_sqrtm(sc, phi, sq):
        return sqrt.propagate_cholesky_factor(phi @ sc, sq)

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df, p, m_pred, e0, e1):
        P0 = e0 @ p
        P1 = e1 @ p
        m_at = P0 @ m_pred
        f = f(t, m_at)
        Jx = df(t, m_at)
        H = P1 - Jx @ P0
        b = Jx @ m_at - f
        z = H @ m_pred + b
        return H, z

    @staticmethod
    def estimate_error(h, sq, z):
        s_sqrtm = h @ sq
        s_chol = sqrt.sqrtm_to_cholesky(s_sqrtm.T)

        whitened_res = jax.scipy.linalg.solve_triangular(s_chol.T, z, lower=False)
        sigma_squared = whitened_res.T @ whitened_res / whitened_res.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error_estimate = sigma * jnp.sqrt(jnp.diag(s_chol @ s_chol.T))
        return error_estimate, sigma


class BatchedEK1(odefilter.ODEFilter):
    """Common functionality for EK1 variations that act on batched multivariate normals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_1d = None
        self.sq_1d = None
        self.batched_sq = None

    def initialize(self, ivp):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        d = self.iwp.wiener_process_dimension
        self.phi_1d, self.sq_1d = self.iwp.preconditioned_discretize_1d

        # No broadcasting possible here (ad-hoc, that is) bc. jax.vmap expects matching batch sizes
        # This can be solved by batching propagate_cholesky_factor differently, but maybe this is not necessary
        self.batched_sq = jnp.stack([self.sq_1d] * d)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_sqrtm = jnp.stack([cov_sqrtm] * d)
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):

        p_1d_raw, p_inv_1d_raw = self.iwp.nordsieck_preconditioner_1d_raw(dt=dt)
        m = p_inv_1d_raw[:, None] * state.y.mean
        sc = p_inv_1d_raw[None, :, None] * state.y.cov_sqrtm

        t = state.t + dt
        new_mean, cov_sqrtm, error, info_dict = self.attempt_unit_step(
            f=state.ivp.f,
            df=state.ivp.df,
            df_diagonal=state.ivp.df_diagonal,
            p_1d_raw=p_1d_raw,
            m=m,
            sc=sc,
            t=t,
        )

        new_mean = p_1d_raw[:, None] * new_mean
        cov_sqrtm = p_1d_raw[None, :, None] * cov_sqrtm

        y1 = jnp.abs(state.y.mean[0])
        y2 = jnp.abs(new_mean[0])
        reference_state = jnp.maximum(y1, y2)

        new_rv = rv.BatchedMultivariateNormal(new_mean, cov_sqrtm)
        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error,
            reference_state=reference_state,
        )
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(m, phi_1d):
        return phi_1d @ m

    @staticmethod
    @jax.jit
    def predict_cov_sqrtm(sc_bd, phi_1d, sq_bd):
        return sqrt.batched_propagate_cholesky_factor(phi_1d @ sc_bd, sq_bd)


class DiagonalEK1(BatchedEK1):
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def attempt_unit_step(self, f, df, df_diagonal, p_1d_raw, m, sc, t):
        m_pred = self.predict_mean(m, phi_1d=self.phi_1d)
        f, Jx_diagonal, z = self.evaluate_ode(
            t=t, f=f, df_diagonal=df_diagonal, p_1d_raw=p_1d_raw, m_pred=m_pred
        )
        error, sigma = self.estimate_error(
            p_1d_raw=p_1d_raw,
            Jx_diagonal=Jx_diagonal,
            sq_bd=self.batched_sq,
            z=z,
        )
        sc_pred = self.predict_cov_sqrtm(
            sc_bd=sc, phi_1d=self.phi_1d, sq_bd=sigma * self.batched_sq
        )
        ss, kgain = self.observe_cov_sqrtm(
            Jx_diagonal=Jx_diagonal, p_1d_raw=p_1d_raw, sc_bd=sc_pred
        )
        cov_sqrtm = self.correct_cov_sqrtm(
            Jx_diagonal=Jx_diagonal,
            p_1d_raw=p_1d_raw,
            sc_bd=sc_pred,
            kgain=kgain,
        )
        new_mean = self.correct_mean(m=m_pred, kgain=kgain, z=z)
        info_dict = dict(num_f_evaluations=1, num_df_diagonal_evaluations=1)
        return new_mean, cov_sqrtm, error, info_dict

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df_diagonal, p_1d_raw, m_pred):
        m_pred_no_precon = p_1d_raw[:, None] * m_pred
        m_at = m_pred_no_precon[0]
        fx = f(t, m_at)
        z = m_pred_no_precon[1] - fx

        Jx_diagonal = df_diagonal(t, m_at)

        return fx, Jx_diagonal, z

    @staticmethod
    @jax.jit
    def estimate_error(p_1d_raw, Jx_diagonal, sq_bd, z):

        sq_bd_no_precon = p_1d_raw[None, :, None] * sq_bd  # shape (d,n,n)
        sq_bd_no_precon_0 = sq_bd_no_precon[:, 0, :]  # shape (d,n)
        sq_bd_no_precon_1 = sq_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sq_bd = (
            sq_bd_no_precon_1 - Jx_diagonal[:, None] * sq_bd_no_precon_0
        )  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sq_bd, h_sq_bd)  # shape (d,)

        xi = z / jnp.sqrt(s)  # shape (d,)
        sigma_squared = xi.T @ xi / xi.shape[0]  # shape ()
        sigma = jnp.sqrt(sigma_squared)  # shape ()
        error_estimate = sigma * jnp.sqrt(s)  # shape (d,)

        return error_estimate, sigma

    @staticmethod
    @jax.jit
    def observe_cov_sqrtm(p_1d_raw, Jx_diagonal, sc_bd):

        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = (
            sc_bd_no_precon_1 - Jx_diagonal[:, None] * sc_bd_no_precon_0
        )  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sc_bd, h_sc_bd)  # shape (d,)
        cross = sc_bd @ h_sc_bd[..., None]  # shape (d,n,1)
        kgain = cross / s[..., None, None]  # shape (d,n,1)

        return jnp.sqrt(s), kgain

    @staticmethod
    @jax.jit
    def correct_cov_sqrtm(p_1d_raw, Jx_diagonal, sc_bd, kgain):
        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = (
            sc_bd_no_precon_1 - Jx_diagonal[:, None] * sc_bd_no_precon_0
        )  # shape (d,n)
        kh_sc_bd = kgain @ h_sc_bd[:, None, :]  # shape (d,n,n)
        new_sc = sc_bd - kh_sc_bd  # shape (d,n,n)
        return new_sc

    @staticmethod
    @jax.jit
    def correct_mean(m, kgain, z):
        correction = kgain @ z[:, None, None]  # shape (d,n,1)
        new_mean = m - correction[:, :, 0].T  # shape (n,d)
        return new_mean


class TruncationEK1(BatchedEK1):
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

    def initialize(self, ivp):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives, wiener_process_dimension=ivp.dimension
        )
        self.P0_1d = self.iwp.projection_matrix_1d(0)
        self.P1_1d = self.iwp.projection_matrix_1d(1)

        d = self.iwp.wiener_process_dimension
        self.P0 = linops.BlockDiagonal(jnp.stack([self.P0_1d] * d))
        self.P1 = linops.BlockDiagonal(jnp.stack([self.P1_1d] * d))

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_sqrtm = jnp.stack([cov_sqrtm] * d)
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
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
        f = state.ivp.f(t, m_at)
        Jx = state.ivp.df(t, m_at)  # Use full Jacobian here!

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
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
        return new_state, info_dict
