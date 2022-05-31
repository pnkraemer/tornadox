"""EK1."""

from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from tornadox.blocks import sqrtutil, stepsizes
from tornadox.blocks.inits import autodiff_first_order
from tornadox.blocks.sde import ibm
from tornadox.blocks.step_impl import ek1_projmatfree_d_nu


def ek1_terminal_value(*, ode_dimension, num_derivatives=5):
    """The traditional EK1, but only for computing the terminal value.

    Uses adaptive steps and PI control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives >= 5,
    and forward-mode initialisation otherweise.

    Uses a dimension-derivative ordering internally,
    and does not use projection matrices.
    """

    EK1State = namedtuple("EK1State", ("u", "dt_proposed", "error_norm", "stats"))

    # Create the identity matrix required for Kronecker-type things below.
    eye_d = jnp.eye(ode_dimension)
    a_1d, q_sqrtm_1d = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    a, q_sqrtm = jnp.kron(eye_d, a_1d), jnp.kron(eye_d, q_sqrtm_1d)

    # Initialisation with autodiff
    if num_derivatives <= 5:
        autodiff_fun = autodiff_first_order.forwardmode_jvp
    else:
        autodiff_fun = autodiff_first_order.taylormode

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
        m0 = m0_mat.reshape((-1,), order="F")
        c_sqrtm0 = jnp.zeros((m0.size, m0.size))
        dt0 = stepsizes.propose_first_dt_per_tol(
            f=f, u0=u0, atol=atol, rtol=rtol, num_derivatives=num_derivatives
        )

        stats = {
            "f_evaluation_count": 0,
            "df_evaluation_count": 0,
            "steps_accepted_count": 0,
            "steps_attempted_count": 0,
            "dt_min": jnp.inf,
            "dt_max": 0.0,
        }
        state = EK1State(u=(m0, c_sqrtm0), dt_proposed=dt0, error_norm=1.0, stats=stats)
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
            p = jnp.tile(p, ode_dimension)
            p_inv = jnp.tile(p_inv, ode_dimension)

            # Attempt step
            x = ek1_projmatfree_d_nu.attempt_step_forward_only(
                f=f,
                df=df,
                m=m0,
                c_sqrtm=c_sqrtm0,
                p=p,
                p_inv=p_inv,
                a=a,
                q_sqrtm=q_sqrtm,
                num_derivatives=num_derivatives,
            )
            (u_proposed, _, error) = x
            error = dt_clipped * error  # The EK1 step does not do this. todo: why not?

            # Normalise the error
            m_proposed, _ = u_proposed
            u1_ref = jnp.maximum(
                jnp.abs(m_proposed[0 :: (num_derivatives + 1)]),
                jnp.abs(m0[0 :: (num_derivatives + 1)]),
            )
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
            dt_new = scale_factor * dt_clipped
            stats = s_prev.stats
            stats["f_evaluation_count"] += 1
            stats["df_evaluation_count"] += 1
            stats["steps_attempted_count"] += 1

            s_new = EK1State(
                u=u_proposed, dt_proposed=dt_new, error_norm=error_norm, stats=stats
            )
            return t_new, s_new

        larger_than_1 = 1.1
        init_val = EK1State(
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

        # Update the statistics
        stats = state.stats
        stats["steps_accepted_count"] += 1
        stats["dt_min"] = jnp.minimum(stats["dt_min"], t - t0)
        stats["dt_max"] = jnp.maximum(stats["dt_max"], t - t0)
        state = EK1State(
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


def ek1_saveat(
    *,
    ode_dimension,
    num_derivatives=5,
):
    """The EK1, but only for computing the checkpoints/saveat-values.

    Uses adaptive steps and PI control.
    Uses time-varying, scalar-valued diffusion.
    Uses Taylor-mode initialisation for num_derivatives > 5,
    and forward-mode initialisation otherwise.
    """

    # Create the identity matrix required for Kronecker-type things below.
    eye_d = jnp.eye(ode_dimension)
    a_1d, q_sqrtm_1d = ibm.system_matrices_1d(num_derivatives=num_derivatives)
    a, q_sqrtm = jnp.kron(eye_d, a_1d), jnp.kron(eye_d, q_sqrtm_1d)

    EK1State = namedtuple(
        "EK1State", ("u", "dt_proposed", "backward_model", "error_norm", "stats")
    )

    # Initialisation with autodiff
    if num_derivatives <= 5:
        autodiff_fun = autodiff_first_order.forwardmode_jvp
    else:
        autodiff_fun = autodiff_first_order.taylormode

    # At each checkpoint, the backward model needs to be 'reset'.
    # This means that it is set to an identity, noise-free transition.
    # We just assemble this once and keep referencing it below.
    identity_transition_matrices = _identity_transition_matrices(
        n=ode_dimension * (num_derivatives + 1)
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
        m0 = m0_mat.reshape((-1,), order="F")
        c_sqrtm0 = jnp.zeros((m0.size, m0.size))
        dt0 = stepsizes.propose_first_dt_per_tol(
            f=f, u0=u0, atol=atol, rtol=rtol, num_derivatives=num_derivatives
        )

        stats = {
            "f_evaluation_count": 0,
            "df_evaluation_count": 0,
            "steps_accepted_count": 0,
            "steps_attempted_count": 0,
            "dt_min": jnp.inf,
            "dt_max": 0.0,
        }

        u0 = (m0, c_sqrtm0)
        s0 = EK1State(
            u=u0,
            backward_model=identity_transition_matrices,
            dt_proposed=dt0,
            error_norm=1.0,
            stats=stats,
        )
        return tspan[0], s0

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
            p = jnp.tile(p, ode_dimension)
            p_inv = jnp.tile(p_inv, ode_dimension)

            # Attempt step
            rv_new, _, _, bw_model, error = ek1_projmatfree_d_nu.attempt_step(
                f=f,
                df=df,
                m=m0,
                c_sqrtm=c_sqrtm0,
                p=p,
                p_inv=p_inv,
                a=a,
                q_sqrtm=q_sqrtm,
                num_derivatives=num_derivatives,
            )
            error = dt_clipped * error  # The solver step does not do this.

            # Normalise the error
            m_new, _ = rv_new
            u_old = m0[0 :: (num_derivatives + 1)]
            u_new = m_new[0 :: (num_derivatives + 1)]
            u1_ref = jnp.abs(jnp.maximum(u_old, u_new))
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
            dt_new = scale_factor * dt_clipped

            stats = s_prev.stats
            stats["f_evaluation_count"] += 1
            stats["df_evaluation_count"] += 1
            stats["steps_attempted_count"] += 1

            s_new = EK1State(
                u=rv_new,
                backward_model=bw_model,
                dt_proposed=dt_new,
                error_norm=error_norm,
                stats=stats,
            )
            return t_new, s_new

        # While-loop until the error_norm is smaller than 1.
        # For the final output, we discard the error_norm.
        larger_than_1 = 1.1
        init_val = EK1State(
            u=state0.u,
            backward_model=state0.backward_model,
            dt_proposed=state0.dt_proposed,
            error_norm=larger_than_1,
            stats=state0.stats,
        )
        t, state = jax.lax.while_loop(
            cond_fun=cond_fun, body_fun=body_fun, init_val=(t0, init_val)
        )

        # Condense both backward models into one.
        # (Technically speaking, this implements 'fixed-point smoothing')
        # todo: if we do this somewhere else, perform_step_fn() can be reused
        #  for other solve_ivp()-style functions.
        A, (b, B_sqrtm) = init_val.backward_model
        C, (d, D_sqrtm) = state.backward_model
        G = A @ C
        xi = A @ d + b
        Xi = sqrtutil.sum_of_sqrtm_factors(S1=A @ D_sqrtm, S2=B_sqrtm)
        backward_model_fp = (G, (xi, Xi))

        stats = state.stats
        stats["steps_accepted_count"] += 1
        stats["dt_min"] = jnp.minimum(stats["dt_min"], t - t0)
        stats["dt_max"] = jnp.maximum(stats["dt_max"], t - t0)
        state = EK1State(
            u=state.u,
            backward_model=backward_model_fp,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=stats,
        )
        return t, state

    @jax.jit
    def reset_state_at_checkpoint_fn(t, state):
        state_reset = EK1State(
            u=state.u,
            backward_model=identity_transition_matrices,
            dt_proposed=state.dt_proposed,
            error_norm=state.error_norm,
            stats=state.stats,
        )
        return t, state_reset

    @jax.jit
    def extract_qoi_fn(t, state):
        return t, state.u, state.backward_model, state.stats

    return init_fn, perform_step_fn, reset_state_at_checkpoint_fn, extract_qoi_fn


@partial(jax.jit, static_argnames=("n",))
def _identity_transition_matrices(*, n):
    a = jnp.eye(n)
    b = jnp.zeros((n,))
    q_sqrtm = jnp.zeros((n, n))
    return a, (b, q_sqrtm)
