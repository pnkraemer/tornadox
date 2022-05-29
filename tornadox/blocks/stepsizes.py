"""Step-size stuff."""

from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnames=("f",))
def propose_first_dt_per_tol(*, f, u0, num_derivatives, rtol, atol):
    # Taken from:
    # https://github.com/google/jax/blob/main/jax/experimental/ode.py
    #
    # which uses the algorithm from
    #
    # E. Hairer, S. P. Norsett G. Wanner,
    # Solving Ordinary Differential Equations I: Nonstiff Problems, Sec. II.4.
    f0 = f(u0)
    scale = atol + u0 * rtol
    a = jnp.linalg.norm(u0 / scale)
    b = jnp.linalg.norm(f0 / scale)
    dt0 = jnp.where((a < 1e-5) | (b < 1e-5), 1e-6, 0.01 * a / b)

    u1 = u0 + dt0 * f0
    f1 = f(u1)
    c = jnp.linalg.norm((f1 - f0) / scale) / dt0
    dt1 = jnp.where(
        (b <= 1e-15) & (c <= 1e-15),
        jnp.maximum(1e-6, dt0 * 1e-3),
        (0.01 / jnp.max(b + c)) ** (1.0 / (num_derivatives + 1)),
    )
    return jnp.minimum(100.0 * dt0, dt1)


@jax.jit
def scale_factor_pi_control(
    *,
    error_norm,
    error_norm_previously_accepted,
    error_order,
    safety,
    factor_min,
    factor_max,
    power_integral_unscaled,
    power_proportional_unscaled,
):
    """Proportional-integral control.

    Proportional-integral control simplifies to integral control
    when the parameters are chosen as

        `power_integral_unscaled=1`,
        `power_proportional_unscaled=0`.
    """
    n1 = power_integral_unscaled / error_order
    n2 = power_proportional_unscaled / error_order

    a1 = (1.0 / error_norm) ** n1
    a2 = (error_norm_previously_accepted / error_norm) ** n2
    scale_factor = safety * a1 * a2

    scale_factor_clipped = jnp.maximum(
        factor_min, jnp.minimum(scale_factor, factor_max)
    )
    return scale_factor_clipped
