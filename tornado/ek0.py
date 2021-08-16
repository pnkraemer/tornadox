import dataclasses
import jax.numpy as jnp

import tornado
from tornado.odesolver import ODESolver


@dataclasses.dataclass
class EK0State:
    ivp: tornado.ivp.InitialValueProblem
    y: jnp.array
    t: float
    error_estimate: jnp.array
    reference_state: jnp.array
    Y: tornado.rv.MultivariateNormal


class EK0(ODESolver):
    def initialize(self, ivp):
        self.d = ivp.dimension
        self.q = self.solver_order
        self.iwp = tornado.iwp.IntegratedWienerTransition(
            wiener_process_dimension=self.d, num_derivatives=self.q
        )
        self.A, self.Ql = self.iwp.preconditioned_discretize_1d
        # A_full, Q_full = iwp.preconditioned_discretize

        Y0_full = tornado.taylor_mode.TaylorModeInitialization()(ivp, self.iwp)
        Y0_kron = tornado.rv.MultivariateNormal(
            Y0_full.mean, jnp.zeros((self.q + 1, self.q + 1))
        )

        self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)
        self.e0 = self.iwp.projection_matrix_1d(0)
        self.e1 = self.iwp.projection_matrix_1d(1)
        self.Id = jnp.eye(self.d)
        self.Iq1 = jnp.eye(self.q + 1)

        return EK0State(
            ivp=ivp,
            y=ivp.y0,
            t=ivp.t0,
            error_estimate=None,
            reference_state=ivp.y0,
            Y=Y0_kron,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        Y = state.Y
        _m, _Cl = Y.mean, Y.cov_cholesky
        A, Ql = self.A, self.Ql

        t_new = state.t + dt

        # [Preconditioners]
        P, PI = self.iwp.nordsieck_preconditioner_1d(dt)
        m, Cl = vec_trick_mul_right(PI, _m), PI @ _Cl

        # [Predict]
        mp = vec_trick_mul_right(A, m)
        Clp = tornado.sqrt.propagate_cholesky_factor(A @ Cl, Ql)

        # [Measure]
        _mp = vec_trick_mul_right(P, mp)  # Undo the preconditioning
        z = self.E1 @ _mp - state.ivp.f(t_new, self.E0 @ _mp)
        H = self.e1 @ P
        Sl = H @ Clp
        S = (Sl @ Sl.T)[0]

        # [Update]
        K = Clp @ Clp.T @ H.T / S
        m_new = m - vec_trick_mul_right(K, z)
        Cl_new = (self.Iq1 - K @ H) @ Clp

        # [Undo preconditioning]
        _m_new, _Cl_new = vec_trick_mul_right(P, m_new), P @ Cl_new

        y_new = self.E0 @ _m_new

        return EK0State(
            ivp=state.ivp,
            y=y_new,
            t=t_new,
            error_estimate=None,
            reference_state=y_new,
            Y=tornado.rv.MultivariateNormal(_m_new, _Cl_new),
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
