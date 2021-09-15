import dataclasses
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg

import tornadox.iwp
from tornadox import init, ivp, iwp, odefilter, rv, sqrt, step
from tornadox.ek1 import BatchedEK1


class ReferenceEK0(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, ivp):

        self.iwp = tornadox.iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        y = rv.MultivariateNormal(
            mean=extended_dy0, cov_sqrtm=jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm)
        )
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        m, Cl = state.y.mean.reshape((-1,), order="F"), state.y.cov_sqrtm
        A, Ql = self.iwp.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, state.ivp.dimension

        # [Predict]
        mp = A @ m

        # Measure / calibrate
        z = self.E1 @ mp - state.ivp.f(state.t + dt, self.E0 @ mp)
        H = self.E1

        S = H @ Ql @ Ql.T @ H.T
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        m_new = mp - K @ z

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict


class KroneckerEK0(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.A = None
        self.Ql = None
        self.e0 = None
        self.e1 = None

    def initialize(self, ivp):
        self.iwp = tornadox.iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.A, self.Ql = self.iwp.preconditioned_discretize_1d

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0
        n, d = self.iwp.num_derivatives + 1, self.iwp.wiener_process_dimension

        self.P0 = self.iwp.projection_matrix(0)
        self.e0 = self.iwp.projection_matrix_1d(0)
        self.e1 = self.iwp.projection_matrix_1d(1)

        y = rv.MatrixNormal(mean=mean, cov_sqrtm_1=jnp.eye(d), cov_sqrtm_2=cov_sqrtm)

        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            y=y,
        )

    @staticmethod
    @jax.jit
    def compute_sigmasquared_error(P, Ql, z):
        _PQl = P[1, 1] * Ql[1, :]
        HQH = _PQl @ _PQl.T
        sigma_squared = z.T @ z / HQH / z.shape[0]
        error_estimate = jnp.stack([jnp.sqrt(sigma_squared * HQH)] * z.shape[0])
        return sigma_squared, error_estimate

    @staticmethod
    @jax.jit
    def update(mp, Clp, P, H, z):
        _PClp = P[1, 1] * Clp[1, :]
        S = _PClp @ _PClp.T
        K = Clp @ (Clp.T @ H.T) / S  # shape (n,1)
        m_new = mp - K * z[None, :]  # shape (n,d)
        Cl_new = Clp - K @ H @ Clp
        return m_new, Cl_new

    @staticmethod
    @partial(jax.jit, static_argnums=(1,))
    def evaluate_ode(t, f, mp, P, e1):
        _mp = P @ mp  # undo the preconditioning
        xi = _mp[0]

        z = _mp[1] - f(t, xi)
        H = e1 @ P
        return z, H

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        Y = state.y
        _m, _Cl = Y.mean, Y.cov_sqrtm_2
        A, Ql = self.A, self.Ql

        t_new = state.t + dt

        # [Preconditioners]
        P, PI = self.iwp.nordsieck_preconditioner_1d(dt)
        m = PI @ _m
        Cl = PI @ _Cl

        # [Predict]
        mp = A @ m

        # [Measure]
        z, H = self.evaluate_ode(t_new, state.ivp.f, mp, P, self.e1)

        # [Calibration]
        sigma_squared, error_estimate = self.compute_sigmasquared_error(P, Ql, z)

        # [Predict Covariance]
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, jnp.sqrt(sigma_squared) * Ql)

        # [Update]
        m_new, Cl_new = self.update(mp, Clp, P, H, z)

        # [Undo preconditioning]
        _m_new = P @ m_new
        _Cl_new = P @ Cl_new

        y_new = jnp.abs(_m_new[0])

        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=t_new,
            error_estimate=error_estimate,
            reference_state=y_new,
            y=rv.MatrixNormal(_m_new, state.y.cov_sqrtm_1, _Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict


class DiagonalEK0(BatchedEK1):
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def attempt_unit_step(self, f, df, df_diagonal, p_1d_raw, m, sc, t):
        m_pred = self.predict_mean(m, phi_1d=self.phi_1d)
        f, z = self.evaluate_ode(
            t=t, f=f, df_diagonal=df_diagonal, p_1d_raw=p_1d_raw, m_pred=m_pred
        )
        error, sigma = self.estimate_error(
            p_1d_raw=p_1d_raw,
            sq_bd=self.batched_sq,
            z=z,
        )
        sc_pred = self.predict_cov_sqrtm(
            sc_bd=sc, phi_1d=self.phi_1d, sq_bd=sigma[:, None, None] * self.batched_sq
        )
        ss, kgain = self.observe_cov_sqrtm(p_1d_raw=p_1d_raw, sc_bd=sc_pred)
        cov_sqrtm = self.correct_cov_sqrtm(
            p_1d_raw=p_1d_raw,
            sc_bd=sc_pred,
            kgain=kgain,
        )
        new_mean = self.correct_mean(m=m_pred, kgain=kgain, z=z)
        info_dict = dict(num_f_evaluations=1)
        return new_mean, cov_sqrtm, error, info_dict

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df_diagonal, p_1d_raw, m_pred):
        m_pred_no_precon = p_1d_raw[:, None] * m_pred
        m_at = m_pred_no_precon[0]
        fx = f(t, m_at)
        z = m_pred_no_precon[1] - fx

        return fx, z

    @staticmethod
    @jax.jit
    def estimate_error(p_1d_raw, sq_bd, z):

        sq_bd_no_precon = p_1d_raw[None, :, None] * sq_bd  # shape (d,n,n)
        sq_bd_no_precon_0 = sq_bd_no_precon[:, 0, :]  # shape (d,n)
        sq_bd_no_precon_1 = sq_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sq_bd = sq_bd_no_precon_1  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sq_bd, h_sq_bd)  # shape (d,)

        xi = z / jnp.sqrt(s)  # shape (d,)
        sigma = jnp.abs(xi)  # shape (d,)
        error_estimate = sigma * jnp.sqrt(s)  # shape (d,)

        return error_estimate, sigma

    @staticmethod
    @jax.jit
    def observe_cov_sqrtm(p_1d_raw, sc_bd):

        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = sc_bd_no_precon_1  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sc_bd, h_sc_bd)  # shape (d,)
        cross = sc_bd @ h_sc_bd[..., None]  # shape (d,n,1)
        kgain = cross / s[..., None, None]  # shape (d,n,1)

        return jnp.sqrt(s), kgain

    @staticmethod
    @jax.jit
    def correct_cov_sqrtm(p_1d_raw, sc_bd, kgain):
        sc_bd_no_precon = p_1d_raw[None, :, None] * sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = sc_bd_no_precon_1  # shape (d,n)
        kh_sc_bd = kgain @ h_sc_bd[:, None, :]  # shape (d,n,n)
        new_sc = sc_bd - kh_sc_bd  # shape (d,n,n)
        return new_sc

    @staticmethod
    @jax.jit
    def correct_mean(m, kgain, z):
        correction = kgain @ z[:, None, None]  # shape (d,n,1)
        new_mean = m - correction[:, :, 0].T  # shape (n,d)
        return new_mean
