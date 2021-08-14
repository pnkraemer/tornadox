"""EK1 solvers."""


import dataclasses

import jax.numpy as jnp

from tornado import ivp, iwp, odesolver, rv, sqrt, taylor_mode


@dataclasses.dataclass
class ODEFilterState:

    ivp: "tornado.ivp.InitialValueProblem"
    t: float
    y: "rv.MultivariateNormal"
    error_estimate: jnp.ndarray
    reference_state: jnp.ndarray


class ReferenceEK1(odesolver.ODESolver):
    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(steprule=steprule, solver_order=num_derivatives)

        # Prior integrated Wiener process
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives, wiener_process_dimension=ode_dimension
        )
        self.P0 = self.iwp.make_projection_matrix(0)
        self.P1 = self.iwp.make_projection_matrix(1)

        # Initialization strategy
        self.tm = taylor_mode.TaylorModeInitialization()

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

        A, SQ = self.iwp.non_preconditioned_discretize(dt)

        m_pred = A @ m
        SC_pred = sqrt.propagate_cholesky_factor(A @ SC, SQ)

        t = state.t + dt
        J = state.ivp.df(t, self.P0 @ m_pred)
        H = self.P1 - J @ self.P0
        z = self.P1 @ m_pred - state.ivp.f(t, self.P0 @ m_pred)

        cov_cholesky, Kgain, sqrt_S = sqrt.update_sqrt(H, SC_pred)
        new_mean = m_pred - Kgain @ z
        new_rv = rv.MultivariateNormal(new_mean, cov_cholesky)
        return ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )
