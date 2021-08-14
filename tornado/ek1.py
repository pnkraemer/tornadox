"""EK1 solvers."""


import dataclasses

import jax.numpy as jnp

import tornado


@dataclasses.dataclass
class ODEFilterState:

    ivp: tornado.ivp.InitialValueProblem
    t: float
    y: tornado.rv.MultivariateNormal
    error_estimate: jnp.ndarray
    reference_state: jnp.ndarray


class EK1(tornado.odesolver.ODESolver):
    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(steprule=steprule, solver_order=num_derivatives)

        # Prior integrated Wiener process
        self.iwp = tornado.iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives, wiener_process_dimension=ode_dimension
        )

        # Initialization strategy
        self.tm = tornado.taylor_mode.TaylorModeInitialization()

    def initialize(self, ivp):
        initial_rv = self.tm(ivp=ivp, prior=self.iwp)
        return ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=initial_rv,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        m, SC = state.y.mean, state.y.cov_cholesky
        P0 = self.iwp.make_projection_matrix(0)
        P1 = self.iwp.make_projection_matrix(1)

        A, SQ = self.iwp.preconditioned_discretize

        m_pred = A @ m
        SC_pred = tornado.sqrt.propagate_cholesky_factor(A @ SC, SQ)

        t = state.t + dt
        J = state.ivp.df(t, P0 @ m_pred)
        H = P1 - J @ P0
        z = P1 @ m_pred - state.ivp.f(t, P0 @ m_pred)

        cov_cholesky, Kgain, sqrt_S = tornado.sqrt.update_sqrt(H, SC_pred)
        new_mean = m_pred - Kgain @ z
        rv = tornado.rv.MultivariateNormal(new_mean, cov_cholesky)
        return ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=rv,
            error_estimate=None,
            reference_state=None,
        )
