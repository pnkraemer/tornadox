"""Kronecker EK0."""

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from tornadox.blocks import stepsizes
from tornadox.blocks.inits import autodiff_first_order
from tornadox.blocks.sde import ibm
from tornadox.blocks.step_impl import ek0_kronecker


def ek0_kronecker_terminal_value(*, num_derivatives=5):
    """The Kronecker EK0, but only for computing the terminal value.

    Uses adaptive steps and proportional control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives >= 5,
    and forward-mode initialisation otherweise.
    """

    # Create the identity matrix required for Kronecker-type things below.
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)

    # Initialisation with autodiff
    if num_derivatives <= 5:
        autodiff_fun = autodiff_first_order.forwardmode_jvp
    else:
        autodiff_fun = autodiff_first_order.taylormode

    KroneckerEK0State = namedtuple(
        "KroneckerEK0State", ("u", "dt_proposed", "error_norm", "stats")
    )

    @partial(jax.jit, static_argnames=("f", "df"))
    def init_fn(
        *,
        f,
        df,
        tspan,
        u0,
        rtol,
        atol,
        safety=0.95,
        factor_min=0.2,
        factor_max=10.0,
        power_integral_unscaled=0.3,
        power_proportional_unscaled=0.4,
    ):

        m0_mat = autodiff_fun(f=f, u0=u0, num_derivatives=num_derivatives)
        c_sqrtm0 = jnp.zeros((num_derivatives + 1, num_derivatives + 1))
        dt0 = stepsizes.propose_first_dt_per_tol(
            f=f, u0=u0, atol=atol, rtol=rtol, num_derivatives=num_derivatives
        )

        stats = {
            "f_evaluation_count": 0,
            "steps_accepted_count": 0,
            "steps_attempted_count": 0,
            "dt_min": jnp.inf,
            "dt_max": 0.0,
        }
        state = KroneckerEK0State(
            u=(m0_mat, c_sqrtm0), dt_proposed=dt0, error_norm=1.0, stats=stats
        )
        return tspan[0], state

    @partial(jax.jit, static_argnames=("f", "df"))
    def perform_step_fn(
        t0,
        state0,
        *,
        f,
        df,
        t1,
        rtol,
        atol,
        safety=0.95,
        factor_min=0.2,
        factor_max=10.0,
        power_integral_unscaled=0.3,
        power_proportional_unscaled=0.4,
    ):
        """Perform a successful step."""

        m0, c_sqrtm0 = state0.u
        error_norm_previously_accepted = state0.error_norm

        @jax.jit
        def cond_fun(s):
            _, state = s
            return state.error_norm > 1

        @jax.jit
        def body_fun(s):
            _, s_prev = s

            # Never exceed the terminal value
            dt_clipped = jnp.minimum(s_prev.dt_proposed, t1 - t0)
            t_new = t0 + dt_clipped

            # Compute preconditioner
            p, p_inv = ibm.preconditioner_diagonal(
                dt=dt_clipped, num_derivatives=num_derivatives
            )

            # Attempt step
            u_new, _, error = ek0_kronecker.attempt_step_forward_only(
                f=f,
                m=m0,
                c_sqrtm=c_sqrtm0,
                p=p,
                p_inv=p_inv,
                a=a,
                q_sqrtm=q_sqrtm,
            )
            error = dt_clipped * error

            # Normalise the error
            m_new, _ = u_new
            u1_ref = jnp.abs(jnp.maximum(m_new[0, :], m0[0, :]))
            error_rel = error / (atol + rtol * u1_ref)
            error_norm = jnp.linalg.norm(error_rel) / jnp.sqrt(error.size)

            # Propose a new time-step
            scale_factor = stepsizes.scale_factor_pi_control(
                error_norm=error_norm,
                error_order=num_derivatives + 1,
                safety=safety,
                factor_min=factor_min,
                factor_max=factor_max,
                error_norm_previously_accepted=error_norm_previously_accepted,
                power_integral_unscaled=power_integral_unscaled,
                power_proportional_unscaled=power_proportional_unscaled,
            )
            dt_proposed = scale_factor * dt_clipped

            stats = s_prev.stats
            stats["f_evaluation_count"] += 1
            stats["steps_attempted_count"] += 1

            state = KroneckerEK0State(
                u=u_new, dt_proposed=dt_proposed, error_norm=error_norm, stats=stats
            )
            return t_new, state

        larger_than_1 = 1.1
        init_val = KroneckerEK0State(
            u=state0.u,
            dt_proposed=state0.dt_proposed,
            error_norm=larger_than_1,
            stats=state0.stats,
        )
        t, state = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=(t0, init_val),
        )
        stats = state.stats
        stats["steps_accepted_count"] += 1
        stats["dt_min"] = jnp.minimum(stats["dt_min"], t - t0)
        stats["dt_max"] = jnp.maximum(stats["dt_max"], t - t0)
        state = KroneckerEK0State(
            u=state.u,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=stats,
        )
        return t, state

    @jax.jit
    def extract_qoi_fn(t, state):
        return t, state.u, state.stats

    return init_fn, perform_step_fn, extract_qoi_fn


@partial(jax.jit, static_argnames=("num_derivatives",))
def _system_matrices_preconditioned(*, eye_d, num_derivatives):
    a, q_sqrtm = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    a = jnp.kron(eye_d, a)
    q_sqrtm = jnp.kron(eye_d, q_sqrtm)
    return a, q_sqrtm
