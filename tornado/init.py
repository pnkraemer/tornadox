from functools import partial

import jax
import jax.numpy as jnp
import scipy.integrate
from jax.experimental.jet import jet

import tornado.iwp
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

    # Force fixed steps via t_eval
    t_eval = jnp.arange(t0, t0 + num_steps * dt, dt)

    # Radau should get the Jacobian if existant
    df = df if df is not None and method == "Radau" else None

    # Compute the data with atol=rtol=1e12 (we want fixed steps!)
    sol = scipy.integrate.solve_ivp(
        fun=f,
        t_span=(t0, t0 + (num_steps - 1) * dt),
        y0=y0,
        atol=1e12,
        rtol=1e12,
        t_eval=t_eval,
        method=method,
        jac=df,
    )
    return sol.t, sol.y.T


def rk_init(t0, num_derivatives, ts, ys):

    d = ys[0].shape[0]
    assert d == 4
    n = num_derivatives + 1
    iwp = tornado.iwp.IntegratedWienerTransition(
        num_derivatives=num_derivatives, wiener_process_dimension=d
    )
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

    filter_res = [(m_loc, sc_loc, None, m0, sc0, None, None, None)]

    # System matrices
    phi_1d, sq_1d = iwp.preconditioned_discretize_1d

    for t, y in zip(ts[1:], ys[1:]):

        # Apply preconditioner (TODO: use raw preconditioner)
        dt = t - t_loc
        p_1d, p_inv_1d = iwp.nordsieck_preconditioner_1d(dt)
        m = p_1d @ m_loc
        sc = p_inv_1d @ sc_loc

        # Predict from t_loc to t
        m_pred = phi_1d @ m
        x = phi_1d @ sc
        sc_pred = tornado.sqrt.propagate_cholesky_factor(x, sq_1d)
        cross = (x @ sc.T).T
        sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

        # Measure and update
        ss = (p_inv_1d @ sc_pred @ sc_pred.T @ p_inv_1d.T)[0, 0]
        kgain = sc_pred @ (sc_pred.T @ p_inv_1d.T)[:, 0] / ss
        z = (p_inv_1d @ m_pred)[0]
        m_loc = m_pred - kgain @ (z - y)
        sc_loc = sc_pred - kgain[:, None] @ (sc_pred[0, :])[None, :]

        # Undo preconditioning
        m = p_inv_1d @ m_loc
        sc = p_inv_1d @ sc_loc

        # Store parameters:
        # (m, sc) are in "normal" coordinates,
        # the others are already preconditioned!
        filter_res.append((m, sc, sgain, m_pred, sc_pred, x, p_1d, p_inv_1d))
        t_loc = t

    # Smoothing pass
    final_out = filter_res[-1]
    m_fut, sc_fut, sgain_fut, m_pred, sc_pref, x, p_1d, p_inv_1d = final_out
    ms, scs = [m_fut], [sc_fut]
    for filter_output in reversed(filter_res[:-1]):

        # Push means and covariances into the preconditioned space
        m, sc = p_1d @ filter_output[0], p_1d @ filter_output[1]
        m_fut, sc_fut = p_1d @ m_fut, p_1d @ sc_fut

        # Make smoothing step
        m_fut, sc_fut = tornado.kalman.smoother_step_sqrt(
            m=m,
            sc=sc,
            m_fut=m_fut,
            sc_fut=sc_fut,
            sgain=sgain_fut,
            sq=sq_1d,
            mp=m_pred,
            x=x,
        )

        # Pull means and covariances back into old coordinates
        # Only for the result of the smoothing step.
        # The other means and covariances are not used anymore.
        m_fut, sc_fut = p_inv_1d @ m_fut, p_inv_1d @ sc_fut

        # Store results
        ms.append(m_fut)
        scs.append(sc_fut)

        # Read out the new parameters
        # They are alreay preconditioned. m_fut, sc_fut are not,
        # but will be pushed into the correct coordinates in the next iteration.
        _, _, sgain_fut, m_pred, sc_pref, x, p_1d, p_inv_1d = filter_output

    return jnp.stack(list(reversed(ms))), jnp.stack(list(reversed(scs)))
