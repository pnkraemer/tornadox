"""Stepsize selection strategies."""

import jax.numpy as jnp

def propose_firststep(ivp):
    norm_y0 = jnp.linalg.norm(ivp.y0)
    norm_dy0 = jnp.linalg.norm(ivp.f(ivp.t0, ivp.y0))
    return 0.01 * norm_y0 / norm_dy0
