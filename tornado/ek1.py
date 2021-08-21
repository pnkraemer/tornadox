"""EK1 solvers."""

from functools import partial

import jax.numpy as jnp
import jax.scipy.linalg

from tornado import iwp, linops, odesolver, rv, sqrt, taylor_mode


class ReferenceEK1(odesolver.ODEFilter):
    """Naive, reference EK1 implementation. Use this to test against."""

    def __init__(self, ode_dimension, steprule, num_derivatives):
        super().__init__(
            ode_dimension=ode_dimension,
            steprule=steprule,
            num_derivatives=num_derivatives,
        )
        self.P0 = self.iwp.projection_matrix(0)
        self.P1 = self.iwp.projection_matrix(1)

    def initialize(self, ivp):
        extended_dy0 = self.tm(
            fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=self.iwp.num_derivatives
        )
        mean = extended_dy0.reshape((-1,), order="F")
        cov_sqrtm = jnp.zeros((mean.shape[0], mean.shape[0]))
        y = rv.MultivariateNormal(mean, cov_sqrtm)
        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        # Extract system matrices
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        P0 = self.P0 @ P
        P1 = self.P1 @ P
        m, SC = Pinv @ state.y.mean, Pinv @ state.y.cov_sqrtm
        A, SQ = self.iwp.preconditioned_discretize

        m_pred = self.predict_mean(m=m, phi=A)

        # Evaluate ODE and create linearisation
        t = state.t + dt
        m_at = P0 @ m_pred
        f = state.ivp.f(t, m_at)
        J = state.ivp.df(t, m_at)
        H = P1 - J @ P0
        b = J @ m_at - f
        z = H @ m_pred + b

        # Calibrate
        sigma, error_estimate = self.calibrate_and_estimate_error(h=H, sq=SQ, z=z)

        # Predict covariance
        SC_pred = self.predict_cov_sqrtm(sc=SC, phi=A, sq=sigma * SQ)

        # Update (observation and correction in one sweep)
        cov_cholesky, Kgain, sqrt_S = sqrt.update_sqrt(H, SC_pred)
        new_mean = m_pred - Kgain @ z

        cov_cholesky = P @ cov_cholesky
        new_mean = P @ new_mean
        new_rv = rv.MultivariateNormal(new_mean, cov_cholesky)

        y1 = jnp.abs(self.P0 @ state.y.mean)
        y2 = jnp.abs(self.P0 @ new_mean)
        reference_state = jnp.maximum(y1, y2)

        # Return new state
        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )

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
    @jax.jit
    def calibrate_and_estimate_error(h, sq, z):
        s_sqrtm = h @ sq
        s_chol = sqrt.sqrtm_to_cholesky(s_sqrtm.T)
        whitened_res = jax.scipy.linalg.solve_triangular(s_chol, z)
        sigma_squared = whitened_res.T @ whitened_res / whitened_res.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error_estimate = sigma * jnp.sqrt(jnp.diag(s_chol @ s_chol.T))
        return sigma, error_estimate


class DiagonalEK1(odesolver.ODEFilter):
    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(
            ode_dimension=ode_dimension,
            steprule=steprule,
            num_derivatives=num_derivatives,
        )

        d = self.iwp.wiener_process_dimension
        self.phi_1d, self.sq_1d = self.iwp.preconditioned_discretize_1d
        self.batched_sq = jnp.stack([self.sq_1d] * d)

    def initialize(self, ivp):
        extended_dy0 = self.tm(
            fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=self.iwp.num_derivatives
        )
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_sqrtm = jnp.zeros((d, n, n))
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odesolver.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=new_rv,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):

        # Todo: make the preconditioners into vectors, not matrices
        p_1d, p_inv_1d = self.iwp.nordsieck_preconditioner_1d(dt=dt)
        m = p_inv_1d @ state.y.mean
        sc = p_inv_1d @ state.y.cov_sqrtm

        t = state.t + dt
        new_mean, cov_sqrtm, error = self.attempt_unit_step(
            f=state.ivp.f, df=state.ivp.df, p_1d=p_1d, m=m, sc=sc, t=t
        )
        new_mean = p_1d @ new_mean
        cov_sqrtm = p_1d @ cov_sqrtm

        y1 = jnp.abs(state.y.mean[0])
        y2 = jnp.abs(new_mean[0])
        reference_state = jnp.maximum(y1, y2)

        new_rv = rv.BatchedMultivariateNormal(new_mean, cov_sqrtm)
        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error,
            reference_state=reference_state,
        )

    @partial(jax.jit, static_argnums=(0, 1, 2))
    def attempt_unit_step(self, f, df, p_1d, m, sc, t):
        m_pred = self.predict_mean(m, phi_1d=self.phi_1d)
        f, J, z = self.evaluate_ode(t=t, f=f, df=df, p_1d=p_1d, m_pred=m_pred)
        error, sigma = self.estimate_error(
            p_1d=p_1d,
            J=J,
            sq_bd=self.batched_sq,
            z=z,
        )
        sc_pred = self.predict_cov_sqrtm(
            sc_bd=sc, phi_1d=self.phi_1d, sq_bd=sigma * self.batched_sq
        )
        ss, kgain = self.observe_cov_sqrtm(J=J, p_1d=p_1d, sc_bd=sc_pred)
        cov_sqrtm = self.correct_cov_sqrtm(
            J=J,
            p_1d=p_1d,
            sc_bd=sc_pred,
            kgain=kgain,
        )
        new_mean = self.correct_mean(m=m_pred, kgain=kgain, z=z)
        return new_mean, cov_sqrtm, error

    @staticmethod
    @jax.jit
    def predict_mean(m, phi_1d):
        return phi_1d @ m

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df, p_1d, m_pred):
        m_pred_no_precon = p_1d @ m_pred
        m_at = m_pred_no_precon[0]
        fx = f(t, m_at)
        Jx = jnp.diag(df(t, m_at))
        z = m_pred_no_precon[1] - fx
        return fx, Jx, z

    @staticmethod
    @jax.jit
    def predict_cov_sqrtm(sc_bd, phi_1d, sq_bd):
        return sqrt.batched_propagate_cholesky_factor(phi_1d @ sc_bd, sq_bd)

    @staticmethod
    @jax.jit
    def estimate_error(p_1d, J, sq_bd, z):

        sq_bd_no_precon = p_1d @ sq_bd  # shape (d,n,n)
        sq_bd_no_precon_0 = sq_bd_no_precon[:, 0, :]  # shape (d,n)
        sq_bd_no_precon_1 = sq_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sq_bd = sq_bd_no_precon_1 - J[:, None] * sq_bd_no_precon_0  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sq_bd, h_sq_bd)  # shape (d,)

        xi = z / jnp.sqrt(s)  # shape (d,)
        sigma_squared = xi.T @ xi / xi.shape[0]  # shape ()
        sigma = jnp.sqrt(sigma_squared)  # shape ()
        error_estimate = sigma * jnp.sqrt(s)  # shape (d,)

        return error_estimate, sigma

    @staticmethod
    @jax.jit
    def observe_cov_sqrtm(p_1d, J, sc_bd):

        sc_bd_no_precon = p_1d @ sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = sc_bd_no_precon_1 - J[:, None] * sc_bd_no_precon_0  # shape (d,n)

        s = jnp.einsum("dn,dn->d", h_sc_bd, h_sc_bd)  # shape (d,)
        cross = sc_bd @ h_sc_bd[..., None]  # shape (d,n,1)
        kgain = cross / s[..., None, None]  # shape (d,n,1)

        return jnp.sqrt(s), kgain

    @staticmethod
    @jax.jit
    def correct_cov_sqrtm(p_1d, J, sc_bd, kgain):
        sc_bd_no_precon = p_1d @ sc_bd  # shape (d,n,n)
        sc_bd_no_precon_0 = sc_bd_no_precon[:, 0, :]  # shape (d,n)
        sc_bd_no_precon_1 = sc_bd_no_precon[:, 1, :]  # shape (d,n)
        h_sc_bd = sc_bd_no_precon_1 - J @ sc_bd_no_precon_0  # shape (d,n)
        kh_sc_bd = kgain @ h_sc_bd[:, None, :]  # shape (d,n,n)
        new_sc = sc_bd - kh_sc_bd  # shape (d,n,n)
        return new_sc

    @staticmethod
    @jax.jit
    def correct_mean(m, kgain, z):
        correction = kgain @ z[:, None, None]  # shape (d,n,1)
        new_mean = m - correction[:, :, 0].T  # shape (n,d)
        return new_mean


class TruncatedEK1(odesolver.ODEFilter):
    """Use full Jacobians for mean-updates, but truncate cleverly to enforce a block-diagonal posterior covariance.

    "Cleverly" means:
    Truncate the Jacobian into a diagonal matrix after the mean update
    and recompute posterior covariance which now has block-diagonal structure (see DiagonalEK1).
    (This also means that for the covariance update, we use the inverse of the diagonal of S, not the diagonal of the inverse of S.)
    """

    def __init__(self, num_derivatives, ode_dimension, steprule):
        super().__init__(
            ode_dimension=ode_dimension,
            steprule=steprule,
            num_derivatives=num_derivatives,
        )

        self.P0_1d = self.iwp.projection_matrix_1d(0)
        self.P1_1d = self.iwp.projection_matrix_1d(1)

        d = self.iwp.wiener_process_dimension
        self.P0 = linops.BlockDiagonal(jnp.stack([self.P0_1d] * d))
        self.P1 = linops.BlockDiagonal(jnp.stack([self.P1_1d] * d))

    def initialize(self, ivp):
        extended_dy0 = self.tm(
            fun=ivp.f, y0=ivp.y0, t0=ivp.t0, num_derivatives=self.iwp.num_derivatives
        )
        d, n = self.iwp.wiener_process_dimension, self.iwp.num_derivatives + 1
        cov_sqrtm = jnp.zeros((d, n, n))
        new_rv = rv.BatchedMultivariateNormal(extended_dy0, cov_sqrtm)
        return odesolver.ODEFilterState(
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
        m = Pinv @ state.y.mean
        SC = Pinv @ state.y.cov_sqrtm
        # assert isinstance(SC, linops.BlockDiagonal)
        # assert SC.array_stack.shape == (d, n, n)

        # Predict [mean]
        m_pred = A @ m

        # Evaluate ODE
        t = state.t + dt
        m_at = P0 @ m_pred
        f = state.ivp.f(t, m_at)
        J = state.ivp.df(t, m_at)  # Use full Jacobian here!

        # Evaluate H @ sth manually (i.e. pseudo-lazily),
        # because P0/P1 slice, and J @ sth is dense matmul.
        b = J @ m_at - f
        z = P1 @ m_pred - J @ (P0 @ m_pred) + b
        # assert isinstance(z, jnp.ndarray)
        # assert z.shape == (d,)

        # Calibrate: Extract P0 @ SC and P1 @ SC
        # only then densify and apply J @ sth.
        SQ0_dense = (P0 @ SQ).todense()
        SQ1_dense = (P1 @ SQ).todense()
        # assert SQ0_dense.shape == (d, d * n)
        # assert SQ1_dense.shape == (d, d * n)
        JSQ0 = J @ SQ0_dense
        # assert JSQ0.shape == (d, d * n)
        S_sqrtm = sqrt.sqrtm_to_cholesky((SQ1_dense - JSQ0).T)
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
        # First project, then apply J (see above)
        SC_pred0_dense = (P0 @ SC_pred).todense()
        SC_pred1_dense = (P1 @ SC_pred).todense()
        # assert SC_pred0_dense.shape == (d, d * n)
        # assert SC_pred1_dense.shape == (d, d * n)
        JSC_pred0 = J @ SC_pred0_dense
        # assert JSC_pred0.shape == (d, d * n)
        S_sqrtm = sqrt.sqrtm_to_cholesky((SC_pred1_dense - JSC_pred0).T)
        # assert S_sqrtm.shape == (d, d)

        # Dense cross-covariance; again, apply P0 and P1 separately from J
        Cminus = SC_pred @ SC_pred.T
        R0 = P0 @ Cminus
        R1 = P1 @ Cminus
        crosscov_transposed = R1.todense() - J @ R0.todense()
        crosscov = crosscov_transposed.T
        # assert crosscov.shape == (d * n, d), crosscov.shape

        # Mean update; contains the only solve() with a dense dxd matrix in the whole script
        # Maybe we turn this into a call to CG at some point
        # (it should be possible to use sparsity of J here; ping @nk for a discussion)
        solved = jax.scipy.linalg.cho_solve((S_sqrtm, True), z)
        new_mean = m_pred - crosscov @ solved
        # assert isinstance(new_mean, jnp.ndarray)
        # assert new_mean.shape == (d * n,)

        # Truncate the hell out of S and K
        # Extract the diagonal from J, and do the rest as in DiagonalEK1.attempt_step()
        # Replicate the respective parts from DiagonalEK1()
        J_as_diag = linops.BlockDiagonal(jnp.diag(J).reshape((-1, 1, 1)))
        H = P1 - J_as_diag @ P0
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
        new_mean = P @ new_mean
        cov_sqrtm = P @ cov_sqrtm
        # assert isinstance(cov_sqrtm, linops.BlockDiagonal)
        # assert cov_sqrtm.array_stack.shape == (d, n, n)

        y1 = jnp.abs(self.P0 @ state.y.mean)
        y2 = jnp.abs(self.P0 @ new_mean)
        reference_state = jnp.maximum(y1, y2)
        # assert isinstance(reference_state, jnp.ndarray)
        # assert reference_state.shape == (d,)
        # assert jnp.all(reference_state >= 0.0), reference_state

        # Return new state
        new_rv = rv.MultivariateNormal(new_mean, cov_sqrtm)
        return odesolver.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
