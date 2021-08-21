import dataclasses

import jax.numpy as jnp

from tornado import ivp, iwp, odesolver, rv, sqrt, step, taylor_mode


class ReferenceEK0(odesolver.ODEFilter):
    def initialize(self, ivp):
        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0 = self.tm(
            fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=self.iwp.num_derivatives
        )
        mean = extended_dy0.reshape((-1,), order="F")
        cov_sqrtm = jnp.zeros((mean.shape[0], mean.shape[0]))
        y = rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)
        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        m, Cl = state.y.mean, state.y.cov_sqrtm
        A, Ql = self.iwp.non_preconditioned_discretize(dt)

        # [Predict]
        mp = A @ m

        # Measure / calibrate
        z = self.E1 @ mp - state.ivp.f(state.t + dt, self.E0 @ mp)
        H = self.E1

        S = H @ Ql @ Ql.T @ H.T
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        print("ref", sigma_squared)

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        m_new = mp - K @ z

        y_new = jnp.abs(self.E0 @ m_new)

        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )


class KroneckerEK0(odesolver.ODEFilter):
    def initialize(self, ivp):

        self.A, self.Ql = self.iwp.preconditioned_discretize_1d

        extended_dy0 = self.tm(
            fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=self.iwp.num_derivatives
        )
        mean = extended_dy0
        n, d = self.iwp.num_derivatives + 1, self.iwp.wiener_process_dimension

        self.P0 = self.iwp.projection_matrix(0)
        self.e0 = self.iwp.projection_matrix_1d(0)
        self.e1 = self.iwp.projection_matrix_1d(1)

        y = rv.MatrixNormal(
            mean=mean, cov_sqrtm_1=jnp.zeros((n, n)), cov_sqrtm_2=jnp.zeros((d, d))
        )

        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            y=y,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        Y = state.y
        _m, _Cl = Y.mean, Y.cov_sqrtm
        A, Ql = self.A, self.Ql

        t_new = state.t + dt

        # [Preconditioners]
        P, PI = self.iwp.nordsieck_preconditioner_1d(dt)
        m, Cl = vec_trick_mul_right(PI, _m), PI @ _Cl

        # [Predict]
        mp = vec_trick_mul_right(A, m)

        # [Measure]
        _mp = vec_trick_mul_right(P, mp)  # Undo the preconditioning
        xi = vec_trick_mul_right(self.e0, _mp)

        z = vec_trick_mul_right(self.e1, _mp) - state.ivp.f(t_new, xi)
        H = self.e1 @ P

        # [Calibration]
        _HQl = H @ Ql
        HQH = _HQl @ _HQl.T  # scalar; to become: HQH = Q11(dt) = Q(dt)[1, 1]
        sigma_squared = z.T @ z / HQH / self.d
        # [Predict Covariance]
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, jnp.sqrt(sigma_squared) * Ql)

        # [Update]
        _HClp = H @ Clp
        S = _HClp @ _HClp.T  # scalar
        K = Clp @ (Clp.T @ H.T) / S
        m_new = mp - vec_trick_mul_right(K, z)
        Cl_new = (self.Iq1 - K @ H) @ Clp

        # [Undo preconditioning]
        _m_new, _Cl_new = vec_trick_mul_right(P, m_new), P @ Cl_new

        y_new = jnp.abs(vec_trick_mul_right(self.e0, _m_new))

        error_estimate = jnp.repeat(jnp.sqrt(sigma_squared * HQH), self.d)

        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=t_new,
            error_estimate=error_estimate,
            reference_state=y_new,
            y=rv.MultivariateNormal(_m_new, _Cl_new),
        )


def vec_trick_mul_full(K1, K2, v):
    """Use the vec trick to compute kron(K1,K2)@v more efficiently"""
    (d1, d2), (d3, d4) = K1.shape, K2.shape
    V = v.reshape(d4, d2, order="F")
    return (K2 @ V @ K1.T).reshape(d1 * d3, order="F")


def vec_trick_mul_right(K2, v):
    """Use the vec trick to compute kron(I_d,K2)@v more efficiently"""
    d3, d4 = K2.shape
    V = v.reshape(d4, v.size // d4, order="F")
    out = K2 @ V
    return out.reshape(out.size, order="F")
