"""Initialization routines."""

import abc
from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
import scipy.integrate
from jax.experimental import ode
from jax.experimental.jet import jet

import tornadox.iwp


class InitializationRoutine(abc.ABC):
    @abc.abstractmethod
    def __call__(self, f, df, y0, t0, num_derivatives):
        raise NotImplementedError


class TaylorMode(InitializationRoutine):

    # Adapter to make it work with ODEFilters
    def __call__(self, f, df, y0, t0, num_derivatives):
        m0 = TaylorMode.taylor_mode(
            fun=f, y0=y0, t0=t0, num_derivatives=num_derivatives
        )
        return m0, jnp.zeros((num_derivatives + 1, num_derivatives + 1))

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    def taylor_mode(fun, y0, t0, num_derivatives):
        """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation."""

        extended_state = jnp.concatenate((jnp.ravel(y0), jnp.array([t0])))
        evaluate_ode_for_extended_state = partial(
            TaylorMode._evaluate_ode_for_extended_state, fun=fun, y0=y0
        )

        # Corner case 1: num_derivatives == 0
        derivs = [y0]
        if num_derivatives == 0:
            return jnp.stack(derivs)

        # Corner case 2: num_derivatives == 1
        initial_series = (jnp.ones_like(extended_state),)
        (
            initial_taylor_coefficient,
            taylor_coefficients,
        ) = TaylorMode.augment_taylor_coefficients(
            evaluate_ode_for_extended_state, extended_state, initial_series
        )
        derivs.append(initial_taylor_coefficient[:-1])
        if num_derivatives == 1:
            return jnp.stack(derivs)

        # Order > 1
        for _ in range(1, num_derivatives):
            _, taylor_coefficients = TaylorMode.augment_taylor_coefficients(
                evaluate_ode_for_extended_state, extended_state, taylor_coefficients
            )
            derivs.append(taylor_coefficients[-2][:-1])
        return jnp.stack(derivs)

    @staticmethod
    def augment_taylor_coefficients(fun, x, taylor_coefficients):
        (init_coeff, [*remaining_taylor_coefficents]) = jet(
            fun=fun,
            primals=(x,),
            series=(taylor_coefficients,),
        )
        taylor_coefficients = (
            init_coeff,
            *remaining_taylor_coefficents,
        )

        return init_coeff, taylor_coefficients

    @staticmethod
    def _evaluate_ode_for_extended_state(extended_state, fun, y0):
        r"""Evaluate the ODE for an extended state (x(t), t).

        More precisely, compute the derivative of the stacked state (x(t), t) according to the ODE.
        This function implements a rewriting of non-autonomous as autonomous ODEs.
        This means that

        .. math:: \dot x(t) = f(t, x(t))

        becomes

        .. math:: \dot z(t) = \dot (x(t), t) = (f(x(t), t), 1).

        Only considering autonomous ODEs makes the jet-implementation
        (and automatic differentiation in general) easier.
        """
        x, t = jnp.reshape(extended_state[:-1], y0.shape), extended_state[-1]
        dx = fun(t, x)
        dx_ravelled = jnp.ravel(dx)
        stacked_ode_eval = jnp.concatenate((dx_ravelled, jnp.array([1.0])))
        return stacked_ode_eval


# RK initialisation


class RungeKutta(InitializationRoutine):
    def __init__(self, dt=0.01, method="RK45", use_df=True):
        self.dt = dt
        self.method = method
        self.stack_initvals = Stack(use_df=use_df)

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt}, method={self.method})"

    def __call__(self, f, df, y0, t0, num_derivatives):
        num_steps = num_derivatives + 1
        ts, ys = self.rk_data(
            f=f, t0=t0, dt=self.dt, num_steps=num_steps, y0=y0, method=self.method
        )
        m, sc = self.stack_initvals(
            f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
        )
        return RungeKutta.rk_init_improve(m=m, sc=sc, t0=t0, ts=ts, ys=ys)

    @staticmethod
    def rk_data(f, t0, dt, num_steps, y0, method):

        # Force fixed steps via t_eval
        t_eval = jnp.arange(t0, t0 + num_steps * dt, dt)

        # Compute the data with atol=rtol=1e12 (we want fixed steps!)
        sol = scipy.integrate.solve_ivp(
            fun=f,
            t_span=(min(t_eval), max(t_eval)),
            y0=y0,
            atol=1e12,
            rtol=1e12,
            t_eval=t_eval,
            method=method,
        )
        return sol.t, sol.y.T

    @staticmethod
    @jax.jit
    def rk_init_improve(m, sc, t0, ts, ys):
        """Improve an initial mean estimate by fitting it to a number of RK steps."""

        d = m.shape[1]
        num_derivatives = m.shape[0] - 1

        # Prior
        iwp = tornadox.iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives, wiener_process_dimension=d // 2
        )
        phi_1d, sq_1d = iwp.preconditioned_discretize_1d

        # Store  -- mean and cov are needed, the other ones should be taken from future steps!
        FilterState = namedtuple(
            "FilterState", "m sc m_pred sc_pred sgain x p_1d_raw p_inv_1d_raw t_loc"
        )

        m_dummy = jnp.nan * m
        sc_dummy = jnp.nan * sc
        p_dummy = jnp.nan * jnp.ones(num_derivatives + 1)
        init_carry_filter = FilterState(
            m=m,
            sc=sc,
            m_pred=m_dummy,
            sc_pred=sc_dummy,
            sgain=sc_dummy,
            x=sc_dummy,
            p_1d_raw=p_dummy,
            p_inv_1d_raw=p_dummy,
            t_loc=t0,
        )

        @jax.jit
        def filter_body_fun(carry, data):
            t, y = data

            # Fetch preconditioner
            t_loc = carry.t_loc
            dt = t - t_loc
            p_1d_raw, p_inv_1d_raw = iwp.nordsieck_preconditioner_1d_raw(dt)

            # Make the next step but return ALL the intermediate quantities
            # (they are needed for efficient smoothing)
            (m, sc, m_pred, sc_pred, sgain, x,) = RungeKutta._forward_filter_step(
                y, carry.sc, carry.m, sq_1d, p_1d_raw, p_inv_1d_raw, phi_1d
            )

            # Store parameters;
            # (m, sc) are in "normal" coordinates, the others are already preconditioned!
            carry = FilterState(
                m=m,
                sc=sc,
                m_pred=m_pred,
                sc_pred=sc_pred,
                sgain=sgain,
                x=x,
                p_1d_raw=p_1d_raw,
                p_inv_1d_raw=p_inv_1d_raw,
                t_loc=t,
            )
            return carry, carry

        # Ignore the first (t,y) pair because this information is already contained in the initial value
        # with certainty, thus it would lead to clashes.
        # The return type of scan() is a single FilterState, where each attribute is stacked (num_t,) times.
        final_out, filter_res = jax.lax.scan(
            filter_body_fun, init_carry_filter, (ts[1:], ys[1:])
        )

        ######### Smoothing #########

        SmoothingState = namedtuple(
            "SmoothingState", "previous_filter_state m_fut sc_fut"
        )

        @jax.jit
        def smoothing_body_fun(carry, filter_state):
            m_fut, sc_fut = carry.m_fut, carry.sc_fut

            bla = carry.previous_filter_state
            m_pred = bla.m_pred
            sgain = bla.sgain
            x = bla.x
            p_1d_raw = bla.p_1d_raw
            p_inv_1d_raw = bla.p_inv_1d_raw

            # _, _, m_pred, _, sgain, x, p_1d_raw, p_inv_1d_raw, _ = carry.previous_filter_state

            # Push means and covariances into the preconditioned space
            m_, sc_ = filter_state.m, filter_state.sc
            m, sc = p_inv_1d_raw[:, None] * m_, p_inv_1d_raw[:, None] * sc_
            m_fut_, sc_fut_ = (
                p_inv_1d_raw[:, None] * m_fut,
                p_inv_1d_raw[:, None] * sc_fut,
            )

            # Make smoothing step
            m_fut__, sc_fut__ = tornadox.kalman.smoother_step_sqrt(
                m=m,
                sc=sc,
                m_fut=m_fut_,
                sc_fut=sc_fut_,
                sgain=sgain,
                sq=sq_1d,
                mp=m_pred,
                x=x,
            )

            # Pull means and covariances back into old coordinates
            # Only for the result of the smoothing step.
            # The other means and covariances are not used anymore.
            m_fut, sc_fut = p_1d_raw[:, None] * m_fut__, p_1d_raw[:, None] * sc_fut__

            # Read out the new parameters
            # They are already preconditioned. m_fut, sc_fut are not,
            # but will be pushed into the correct coordinates in the next iteration.
            new_carry = SmoothingState(
                previous_filter_state=filter_state, m_fut=m_fut, sc_fut=sc_fut
            )
            return new_carry, (m_fut, sc_fut)

        # Remove the final time point from the data -- no smoothing necessary here
        data_ = FilterState(*[x[:-1] for x in filter_res])

        # Run the backwards loop
        init_carry_smoother = SmoothingState(
            previous_filter_state=final_out, m_fut=final_out.m, sc_fut=final_out.sc
        )
        almost_done, _ = jax.lax.scan(
            smoothing_body_fun, init_carry_smoother, data_, reverse=True
        )
        _, (m, sc) = smoothing_body_fun(almost_done, init_carry_filter)
        return m, sc

    @staticmethod
    @jax.jit
    def _forward_filter_step(y, sc, m, sq_1d, p_1d_raw, p_inv_1d_raw, phi_1d):

        # Apply preconditioner
        m = p_inv_1d_raw[:, None] * m
        sc = p_inv_1d_raw[:, None] * sc

        # Predict from t_loc to t
        m_pred = phi_1d @ m
        x = phi_1d @ sc
        sc_pred = tornadox.sqrt.propagate_cholesky_factor(x, sq_1d)

        # Compute the gainzz
        cross = (x @ sc.T).T
        sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

        # Measure (H := "slicing" \circ "remove preconditioner")
        sc_pred_np = p_1d_raw[:, None] * sc_pred
        h_sc_pred = sc_pred_np[0, :]
        s = h_sc_pred @ h_sc_pred.T
        cross = sc_pred @ h_sc_pred.T
        kgain = cross / s
        z = (p_1d_raw[:, None] * m_pred)[0]

        # Update (with a good sprinkle of broadcasting)
        m_loc = m_pred - kgain[:, None] * (z - y)[None, :]
        sc_loc = sc_pred - kgain[:, None] * h_sc_pred[None, :]

        # Undo preconditioning
        m = p_1d_raw[:, None] * m_loc
        sc = p_1d_raw[:, None] * sc_loc

        return m, sc, m_pred, sc_pred, sgain, x


class CompiledRungeKutta(RungeKutta):
    def __init__(self, dt=0.01, method="RK45", use_df=True):
        if method != "RK45":
            raise ValueError("CompiledRungeKutta does RK45 only.")
        super().__init__(dt=dt, method=method, use_df=use_df)

    # Repeat the implementation from above, but this time, we can use the jax.jit decorator.
    @partial(jax.jit, static_argnums=(0, 1, 2, 5))
    def __call__(self, f, df, y0, t0, num_derivatives):
        num_steps = num_derivatives + 1
        ts, ys = self.rk_data(
            f=f, t0=t0, dt=self.dt, num_steps=num_steps, y0=y0, method=self.method
        )
        m, sc = self.stack_initvals(
            f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
        )
        return RungeKutta.rk_init_improve(m=m, sc=sc, t0=t0, ts=ts, ys=ys)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 3, 5))
    def rk_data(f, t0, dt, num_steps, y0, method):
        # Generate RK data via jax.experimental.ode

        # "data" is unused, because we loop forward in time, and not _over_ something.
        # https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html#jax.lax.scan
        def body_fun_rk(state, data, func, dt):
            t, y, fy = state
            y_next, fy_next, *_ = ode.runge_kutta_step(func, y, fy, t, dt)
            t_next = t + dt
            return (t_next, y_next, fy_next), (t_next, y_next)

        # jax.experimental.ode wants different signatures
        f_reversed_inputs = lambda y, t: f(t, y)
        body_fun = jax.jit(partial(body_fun_rk, func=f_reversed_inputs, dt=dt))
        init_state = (t0, y0, f(t0, y0))
        _, (ts, ys) = jax.lax.scan(
            f=body_fun, init=init_state, xs=None, length=num_steps
        )
        return ts, ys


class Stack(InitializationRoutine):
    def __init__(self, use_df=True):
        if use_df:
            self.call_init = Stack.initial_state_jac
        else:
            self.call_init = Stack.initial_state_no_jac

    @partial(jax.jit, static_argnums=(0, 1, 2, 5))
    def __call__(self, f, df, y0, t0, num_derivatives):
        return self.call_init(f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives)

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 4))
    def initial_state_jac(f, df, y0, t0, num_derivatives):
        d = y0.shape[0]
        n = num_derivatives + 1

        fy = f(t0, y0)
        dfy = df(t0, y0)
        m = jnp.stack([y0, fy, dfy @ fy] + [jnp.zeros(d)] * (n - 3))
        sc = jnp.diag(jnp.array([0.0, 0.0, 0.0] + [1e3] * (n - 3)))
        return m, sc

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 4))
    def initial_state_no_jac(f, df, y0, t0, num_derivatives):
        d = y0.shape[0]
        n = num_derivatives + 1

        fy = f(t0, y0)
        m = jnp.stack([y0, fy] + [jnp.zeros(d)] * (n - 2))
        sc = jnp.diag(jnp.array([0.0, 0.0] + [1e3] * (n - 2)))
        return m, sc
