import dataclasses

import jax.numpy as jnp

import tornadox.iwp
from tornadox import init, ivp, iwp, odefilter, rv, sqrt, step


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
        mean = extended_dy0  # .reshape((-1,), order="F")
        y = rv.MultivariateNormal(
            mean=mean, cov_sqrtm=jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm)
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

        return odefilter.ODEFilterState(
            ivp=state.ivp,
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )


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
        _mp = P @ mp  # undo the preconditioning
        xi = _mp[0]

        z = _mp[1] - state.ivp.f(t_new, xi)
        H = self.e1 @ P

        # [Calibration]
        HQH = (P @ Ql @ Ql.T @ P.T)[1, 1]
        sigma_squared = z.T @ z / HQH / z.shape[0]

        # [Predict Covariance]
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, jnp.sqrt(sigma_squared) * Ql)

        # [Update]
        S = (P @ Clp @ Clp.T @ P.T)[1, 1]
        K = Clp @ (Clp.T @ H.T) / S  # shape (n,1)
        m_new = mp - K * z[None, :]  # shape (n,d)
        Cl_new = Clp - K @ H @ Clp

        # [Undo preconditioning]
        _m_new = P @ m_new
        _Cl_new = P @ Cl_new

        y_new = jnp.abs(_m_new[0])

        d = z.shape[0]
        error_estimate = jnp.stack([jnp.sqrt(sigma_squared * HQH)] * d)

        return odefilter.ODEFilterState(
            ivp=state.ivp,
            t=t_new,
            error_estimate=error_estimate,
            reference_state=y_new,
            y=rv.MatrixNormal(_m_new, state.y.cov_sqrtm_1, _Cl_new),
        )
