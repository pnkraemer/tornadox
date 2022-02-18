"""EK1 solvers."""

from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg

from tornadox import iwp, odefilter, rv, sqrt


class ReferenceEK1(odefilter.ODEFilter):
    """Naive, reference EK1 implementation. Use this to test against."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.P1 = None

    def initialize(self, f, t0, tmax, y0, df, df_diagonal):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=y0.shape[0],
        )
        self.P0 = self.iwp.projection_matrix(0)
        self.P1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=f,
            df=df,
            y0=y0,
            t0=t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0  # .reshape((-1,), order="F")
        y = rv.MultivariateNormal(mean, jnp.kron(jnp.eye(y0.shape[0]), cov_sqrtm))
        return odefilter.ODEFilterState(
            t=t0,
            y=y,
            error_estimate=jnp.nan * jnp.ones(self.iwp.wiener_process_dimension),
            reference_state=jnp.nan * jnp.ones(self.iwp.wiener_process_dimension),
        )

    @partial(jax.jit, static_argnums=(0, 3, 7, 8))
    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):
        # Extract system matrices
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, SQ = self.iwp.preconditioned_discretize
        t = state.t + dt
        n, d = self.num_derivatives + 1, self.iwp.wiener_process_dimension

        # Pull states into preconditioned state
        m, SC = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        cov_cholesky, error_estimate, new_mean = self.attempt_unit_step(
            A, P, SC, SQ, m, state, t, f, t0, tmax, y0, df, df_diagonal
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
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
        return new_state, info_dict

    @partial(jax.jit, static_argnums=(0, 8, 12, 13))
    def attempt_unit_step(
        self, A, P, SC, SQ, m, state, t, f, t0, tmax, y0, df, df_diagonal
    ):
        m_pred = self.predict_mean(m=m, phi=A)
        H, z = self.evaluate_ode(
            t=t,
            f=f,
            df=df,
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


class ReferenceEK1ConstantDiffusion(ReferenceEK1):
    """Can only be used with jax.disable_jit()."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.diffusion_list_scalar = []

    def solve(self, *args, **kwargs):
        raise RuntimeError(
            "I cannot do 'solve()'. Do 'simulate_final_state()' instead."
        )

    def simulate_final_state(self, *args, **kwargs):
        final_state, info = super().simulate_final_state(*args, **kwargs)
        s = jnp.sqrt(jnp.mean(jnp.asarray(self.diffusion_list_scalar)))
        final_state = final_state._replace(
            y=final_state.y._replace(cov_sqrtm=s * final_state.y.cov_sqrtm)
        )
        return final_state, info

    # Does not like JIT bc side effects (append).
    # Use with caution.
    def attempt_unit_step(
        self, A, P, SC, SQ, m, state, t, f, t0, tmax, y0, df, df_diagonal
    ):
        m_pred = self.predict_mean(m=m, phi=A)
        H, z = self.evaluate_ode(
            t=t,
            f=f,
            df=df,
            p=P,
            m_pred=m_pred,
            e0=self.P0,
            e1=self.P1,
        )
        error_estimate, _ = self.estimate_error(h=H, sq=SQ, z=z)
        SC_pred = self.predict_cov_sqrtm(sc=SC, phi=A, sq=SQ)
        cov_cholesky, Kgain, sqrt_S = sqrt.update_sqrt(H, SC_pred)
        new_mean = m_pred - Kgain @ z

        diff = z @ jnp.linalg.solve(sqrt_S @ sqrt_S.T, z) / z.shape[0]
        self.diffusion_list_scalar.append(diff)
        return cov_cholesky, error_estimate, new_mean


class BatchedEK1(odefilter.ODEFilter):
    """Common functionality for EK1 variations that act on batched multivariate normals."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_1d = None
        self.sq_1d = None
        self.batched_sq = None

    def initialize(self, f, t0, tmax, y0, df, df_diagonal):
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=y0.shape[0],
        )
        self.phi_1d, self.batched_sq = self.create_system()
        extended_dy0, cov_sqrtm = self.init(
            f=f,
            df=df,
            y0=y0,
            t0=t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        return BatchedEK1.y0_to_initial_state(
            extended_dy0, cov_sqrtm, d=y0.shape[0], t0=t0
        )

    @partial(jax.jit, static_argnums=0)
    def create_system(self):
        phi_1d, sq_1d = self.iwp.preconditioned_discretize_1d

        # No broadcasting possible here (ad-hoc, that is) bc. jax.vmap expects matching batch sizes
        # This can be solved by batching propagate_cholesky_factor differently, but maybe this is not necessary
        batched_sq = jnp.stack([sq_1d] * self.iwp.wiener_process_dimension)
        return phi_1d, batched_sq

    @staticmethod
    @partial(jax.jit, static_argnums=2)
    def y0_to_initial_state(extended_dy0, cov_sqrtm, d, t0):
        cov_sqrtm = jnp.stack([cov_sqrtm] * d)
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odefilter.ODEFilterState(
            t=t0,
            y=new_rv,
            error_estimate=jnp.nan * jnp.ones(d),
            reference_state=jnp.nan * jnp.ones(d),
        )

    @partial(jax.jit, static_argnums=(0, 3, 7, 8))
    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):

        p_1d_raw, p_inv_1d_raw = self.iwp.nordsieck_preconditioner_1d_raw(dt=dt)
        m = p_inv_1d_raw[:, None] * state.y.mean
        sc = p_inv_1d_raw[None, :, None] * state.y.cov_sqrtm

        t = state.t + dt
        new_mean, cov_sqrtm, error, info_dict = self.attempt_unit_step(
            f=f,
            df=df,
            df_diagonal=df_diagonal,
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
            sc_bd=sc, phi_1d=self.phi_1d, sq_bd=sigma[:, None, None] * self.batched_sq
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
        sigma = jnp.abs(xi)  # shape (d,)
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
        s += 1e-16
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
