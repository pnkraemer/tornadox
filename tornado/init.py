from functools import partial

import jax
import jax.numpy as jnp
import scipy.integrate
from jax.experimental.jet import jet

from tornado import rv


def taylor_mode(fun, y0, t0, num_derivatives):
    """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation."""

    extended_state = jnp.concatenate((jnp.ravel(y0), jnp.array([t0])))
    evaluate_ode_for_extended_state = partial(
        _evaluate_ode_for_extended_state, fun=fun, y0=y0
    )

    # Corner case 1: num_derivatives == 0
    derivs = [y0]
    if num_derivatives == 0:
        return jnp.stack(derivs)

    # Corner case 2: num_derivatives == 1
    initial_series = (jnp.ones_like(extended_state),)
    (initial_taylor_coefficient, [*remaining_taylor_coefficents]) = jet(
        fun=evaluate_ode_for_extended_state,
        primals=(extended_state,),
        series=(initial_series,),
    )
    derivs.append(initial_taylor_coefficient[:-1])
    if num_derivatives == 1:
        return jnp.stack(derivs)

    # Order > 1
    for _ in range(1, num_derivatives):
        taylor_coefficients = (
            initial_taylor_coefficient,
            *remaining_taylor_coefficents,
        )
        (_, [*remaining_taylor_coefficents]) = jet(
            fun=evaluate_ode_for_extended_state,
            primals=(extended_state,),
            series=(taylor_coefficients,),
        )
        derivs.append(remaining_taylor_coefficents[-2][:-1])
    return jnp.stack(derivs)


# @partial(jax.jit, static_argnums=(1,))
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


def rk_data(f, t0, dt, num_steps, y0, method, df=None):
    t_eval = jnp.arange(t0, t0 + (num_steps + 1) * dt, dt)
    df = df if df is not None and method == "Radau" else None
    sol = scipy.integrate.solve_ivp(
        f,
        (t0, t0 + (num_steps + 1) * dt),
        y0=y0,
        atol=1e-12,
        rtol=1e-12,
        t_eval=t_eval,
        method=method,
        jac=df,
    )
    return sol.t, sol.y.T


def rk_init(t0, num_derivatives, ts, ys):

    d = ys[0].shape[0]
    assert d == 4
    n = num_derivatives + 1

    # Initial mean and cov
    m0 = jnp.zeros((n, d))
    sc0 = 1e4 * jnp.eye(n)

    # Initial update:
    ss = sc0[0, 0]
    kgain = sc0[:, 0] / (ss ** 2)
    z = m0[0]

    m_loc = m0 - kgain @ (z - ys[0])
    sc_loc = sc0 - kgain[:, None] @ (sc0[0, :])[None, :]
    t_loc = t0
    ms, scs = [m_loc], [sc_loc]

    for t, y in zip(ts[1:], ys[1:]):

        # Predict from t_loc to t
        dt = t - t_loc

    return jnp.stack(ms), jnp.stack(scs)
