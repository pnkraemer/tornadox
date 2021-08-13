import jax.numpy as jnp
from jax.experimental.jet import jet

from tornado import rv


class TaylorModeInitialization:
    """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation."""

    def __call__(self, ivp, prior) -> rv.MultivariateNormal:

        num_derivatives = prior.num_derivatives

        dt = jnp.array([1.0])

        def evaluate_ode_for_extended_state(extended_state, ivp=ivp, dt=dt):
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
            x, t = jnp.reshape(extended_state[:-1], ivp.y0.shape), extended_state[-1]
            dx = ivp.f(t, x)
            dx_ravelled = jnp.ravel(dx)
            stacked_ode_eval = jnp.concatenate((dx_ravelled, dt))
            return stacked_ode_eval

        def derivs_to_normal_randvar(derivs):
            """Finalize the output in terms of creating a suitably sized random
            variable."""
            all_derivs = prior.reorder_state_from_derivative_to_coordinate(
                jnp.asarray(derivs)
            )

            return rv.MultivariateNormal(
                mean=jnp.asarray(all_derivs),
                cov_cholesky=jnp.asarray(jnp.diag(jnp.zeros(len(derivs)))),
            )

        extended_state = jnp.concatenate((jnp.ravel(ivp.y0), jnp.array([ivp.t0])))
        derivs = []

        # Corner case 1: num_derivatives == 0
        derivs.extend(ivp.y0)
        if num_derivatives == 0:
            return derivs_to_normal_randvar(derivs=derivs)

        # Corner case 2: num_derivatives == 1
        initial_series = (jnp.ones_like(extended_state),)
        (initial_taylor_coefficient, [*remaining_taylor_coefficents]) = jet(
            fun=evaluate_ode_for_extended_state,
            primals=(extended_state,),
            series=(initial_series,),
        )
        derivs.extend(initial_taylor_coefficient[:-1])
        if num_derivatives == 1:
            return derivs_to_normal_randvar(derivs=derivs)

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
            derivs.extend(remaining_taylor_coefficents[-2][:-1])
        return derivs_to_normal_randvar(derivs=derivs)
