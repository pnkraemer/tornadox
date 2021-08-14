"""EK1 solvers."""

import dataclasses

import jax.numpy as jnp

from tornado import ivp, iwp, linops, odesolver, rv, sqrt, taylor_mode


@dataclasses.dataclass
class ODEFilterState:

    ivp: "tornado.ivp.InitialValueProblem"
    t: float
    y: "rv.MultivariateNormal"
    error_estimate: jnp.ndarray
    reference_state: jnp.ndarray


class ReferenceEK1(odesolver.ODESolver):
    """Naive, reference EK1 implementation. Use this to test against."""

    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(steprule=steprule, solver_order=num_derivatives)

        # Prior integrated Wiener process
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives, wiener_process_dimension=ode_dimension
        )
        self.P0 = self.iwp.projection_matrix(0)
        self.P1 = self.iwp.projection_matrix(1)

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
        # Extract system matrices
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        P0 = self.P0 @ P
        P1 = self.P1 @ P
        m, SC = Pinv @ state.y.mean, Pinv @ state.y.cov_cholesky
        A, SQ = self.iwp.preconditioned_discretize

        # Prediction
        m_pred = A @ m
        SC_pred = sqrt.propagate_cholesky_factor(A @ SC, SQ)

        # Evaluate ODE
        t = state.t + dt
        m_at = P0 @ m_pred
        f = state.ivp.f(t, m_at)
        J = state.ivp.df(t, m_at)

        # Create linearisation
        H = P1 - J @ P0
        b = J @ m_at - f

        # Update
        cov_cholesky, Kgain, sqrt_S = sqrt.update_sqrt(H, SC_pred)
        z = H @ m_pred + b
        new_mean = m_pred - Kgain @ z

        cov_cholesky = P @ cov_cholesky
        new_mean = P @ new_mean
        new_rv = rv.MultivariateNormal(new_mean, cov_cholesky)

        # Return new state
        return ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )


class DiagonalEK1(odesolver.ODESolver):
    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(steprule=steprule, solver_order=num_derivatives)

        # Prior integrated Wiener process
        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives, wiener_process_dimension=ode_dimension
        )
        self.P0_1d = self.iwp.projection_matrix_1d(0)
        self.P1_1d = self.iwp.projection_matrix_1d(1)

        d = self.iwp.wiener_process_dimension
        self.P0 = linops.BlockDiagonal(jnp.stack([self.P0_1d] * d))
        self.P1 = linops.BlockDiagonal(jnp.stack([self.P1_1d] * d))

        # Initialization strategy
        self.tm = taylor_mode.TaylorModeInitialization()

    def initialize(self, ivp):
        initial_rv = self.tm(ivp=ivp, prior=self.iwp)
        mean = initial_rv.mean
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_cholesky = linops.BlockDiagonal(array_stack=jnp.zeros((d, n, n)))
        new_rv = rv.MultivariateNormal(mean, cov_cholesky)
        return ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        # Extract system matrices
        d = self.iwp.wiener_process_dimension
        n = self.iwp.num_derivatives + 1
        P, Pinv = self.iwp.nordsieck_preconditioner_1d(dt)
        m, SC = state.y.mean, state.y.cov_cholesky
        A, SQ = self.iwp.preconditioned_discretize_1d
        A = P @ A @ Pinv
        SQ = P @ SQ

        A = linops.BlockDiagonal(jnp.stack([A] * d))
        SQ = linops.BlockDiagonal(jnp.stack([SQ] * d))
        assert isinstance(A, linops.BlockDiagonal)
        assert isinstance(SQ, linops.BlockDiagonal)
        assert A.array_stack.shape == (d, n, n)
        assert SQ.array_stack.shape == (d, n, n)

        # Prediction
        m_pred = A @ m
        SC_pred = linops.BlockDiagonal(
            jnp.stack(
                [
                    sqrt.propagate_cholesky_factor(a @ sc, sq)
                    for (a, sc, sq) in zip(
                        A.array_stack, SC.array_stack, SQ.array_stack
                    )
                ]
            )
        )
        assert isinstance(SC_pred, linops.BlockDiagonal)
        assert SC_pred.array_stack.shape == (d, n, n)

        # Evaluate ODE
        t = state.t + dt
        m_at = self.P0 @ m_pred
        f = state.ivp.f(t, m_at)
        J = state.ivp.df(t, m_at)
        diag_J = linops.BlockDiagonal(jnp.diag(J).reshape((-1, 1, 1)))
        assert isinstance(diag_J, linops.BlockDiagonal)
        assert diag_J.array_stack.shape == (d, 1, 1)

        # Create linearisation
        H = self.P1 - diag_J @ self.P0
        b = J @ m_at - f
        assert isinstance(H, linops.BlockDiagonal)
        assert H.array_stack.shape == (d, 1, n)

        # Update
        cov_cholesky = linops.BlockDiagonal(
            jnp.stack(
                [
                    sqrt.update_sqrt(h, sc_pred)[0]
                    for (h, sc_pred) in zip(H.array_stack, SC_pred.array_stack)
                ]
            )
        )
        assert isinstance(cov_cholesky, linops.BlockDiagonal)
        assert cov_cholesky.array_stack.shape == (d, n, n)

        Kgain = linops.BlockDiagonal(
            jnp.stack(
                [
                    sqrt.update_sqrt(h, sc_pred)[1]
                    for (h, sc_pred) in zip(H.array_stack, SC_pred.array_stack)
                ]
            )
        )
        assert isinstance(Kgain, linops.BlockDiagonal)
        assert Kgain.array_stack.shape == (d, n, 1)

        z = H @ m_pred + b
        new_mean = m_pred - Kgain @ z
        new_rv = rv.MultivariateNormal(new_mean, cov_cholesky)

        # Return new state
        return ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )
