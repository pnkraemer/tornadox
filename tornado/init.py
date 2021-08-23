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
        t_span=(min(t_eval), max(t_eval)),
        y0=y0,
        atol=1e12,
        rtol=1e12,
        t_eval=t_eval,
        method=method,
        jac=df,
    )
    return sol.t, sol.y.T


# todo:
# phrase this as "improve m0", which already knows num_derivatives and dim,
# and does not need to see f, df, y0
# signature: rk_improve(m0, t0, ts, ys)
# respectively rk_improve(m0, c0, t0, ts, ys)
# which can then also be run in an EM style.
def rk_init(f, df, y0, t0, num_derivatives, ts, ys):

    d = ys[0].shape[0]
    iwp = tornado.iwp.IntegratedWienerTransition(
        num_derivatives=num_derivatives, wiener_process_dimension=d // 2
    )

    # System matrices
    phi_1d, sq_1d = iwp.preconditioned_discretize_1d

    # Initial mean and cov
    n = num_derivatives + 1
    m0 = jnp.zeros((n, d))
    sc0 = 1e2 * jnp.eye(n)

    # Initial update: (no preconditioning, conditioning is fine here)
    s = (sc0 @ sc0.T)[0, 0]
    kgain = (sc0 @ sc0.T)[:, 0] / s

    # Observe
    z = m0[0]
    ys0 = ys[0]

    # Correct
    m = m0 - kgain[:, None] @ (z - ys0)[None, :]
    sc = sc0 - kgain[:, None] @ (sc0[0, :])[None, :]

    fy = f(t0, y0)
    dfy = df(t0, y0)
    m = jnp.stack([y0, fy, dfy @ fy] + [jnp.zeros(d)] * (n - 3))
    sc = jnp.diag(jnp.array([0.0, 0.0, 0.0] + [1.0] * (n - 3)))

    # Store
    filter_res = [(m, sc, None, None, None, None, None, None)]
    t_loc = t0

    for t, y in zip(ts[1:], ys[1:]):

        # Fetch preconditioner
        dt = t - t_loc
        p_1d_raw, p_inv_1d_raw = iwp.nordsieck_preconditioner_1d_raw(dt)

        # Apply preconditioner
        m = p_inv_1d_raw[:, None] * m
        sc = p_inv_1d_raw[:, None] * sc

        # Predict from t_loc to t
        m_pred = phi_1d @ m
        x = phi_1d @ sc
        sc_pred = tornado.sqrt.propagate_cholesky_factor(x, sq_1d)

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

        m_loc = m_pred - kgain[:, None] * (z - y)[None, :]
        sc_loc = sc_pred - kgain[:, None] * h_sc_pred[None, :]

        # Undo preconditioning
        m = p_1d_raw[:, None] * m_loc
        sc = p_1d_raw[:, None] * sc_loc

        # Store parameters:
        # (m, sc) are in "normal" coordinates,
        # the others are already preconditioned!
        filter_res.append((m, sc, sgain, m_pred, sc_pred, x, p_1d_raw, p_inv_1d_raw))
        t_loc = t

    # Smoothing pass
    final_out = filter_res[-1]
    m_fut, sc_fut, sgain_fut, m_pred, _, x, p_1d_raw, p_inv_1d_raw = final_out

    for filter_output in reversed(filter_res[:-1]):

        # Push means and covariances into the preconditioned space
        m_, sc_ = filter_output[0], filter_output[1]
        m, sc = p_inv_1d_raw[:, None] * m_, p_inv_1d_raw[:, None] * sc_
        m_fut_, sc_fut_ = p_inv_1d_raw[:, None] * m_fut, p_inv_1d_raw[:, None] * sc_fut

        # Make smoothing step
        m_fut__, sc_fut__ = tornado.kalman.smoother_step_sqrt(
            m=m,
            sc=sc,
            m_fut=m_fut_,
            sc_fut=sc_fut_,
            sgain=sgain_fut,
            sq=sq_1d,
            mp=m_pred,
            x=x,
        )

        # Pull means and covariances back into old coordinates
        # Only for the result of the smoothing step.
        # The other means and covariances are not used anymore.
        m_fut, sc_fut = p_1d_raw[:, None] * m_fut__, p_1d_raw[:, None] * sc_fut__

        # Read out the new parameters
        # They are alreay preconditioned. m_fut, sc_fut are not,
        # but will be pushed into the correct coordinates in the next iteration.
        _, _, sgain_fut, m_pred, _, x, p_1d_raw, p_inv_1d_raw = filter_output

    return m_fut, sc_fut
