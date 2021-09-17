"""Stepsize selection strategies."""

import abc
from functools import partial

import jax
import jax.numpy as jnp


class StepRule(abc.ABC):
    """Step-size selection rules for ODE solvers."""

    @abc.abstractmethod
    def suggest(self, previous_dt, scaled_error_estimate, local_convergence_rate=None):
        raise NotImplementedError

    @abc.abstractmethod
    def is_accepted(self, scaled_error_estimate):
        raise NotImplementedError

    def scale_error_estimate(self, unscaled_error_estimate, reference_state):
        raise NotImplementedError

    def first_dt(self, ivp):
        raise NotImplementedError


class ConstantSteps(StepRule):
    """Constant step-sizes."""

    def __init__(self, dt):
        self.dt = dt

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt})"

    @partial(jax.jit, static_argnums=0)
    def suggest(self, previous_dt, scaled_error_estimate, local_convergence_rate=None):
        return self.dt

    @partial(jax.jit, static_argnums=0)
    def is_accepted(self, scaled_error_estimate):
        return True

    @partial(jax.jit, static_argnums=0)
    def scale_error_estimate(self, unscaled_error_estimate, reference_state):
        # Return None to make sure this quantity is not used further below
        return None

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def first_dt(self, f, t0, tmax, y0, df, df_diagonal):
        return self.dt


class AdaptiveSteps(StepRule):
    def __init__(
        self,
        abstol=1e-4,
        reltol=1e-2,
        max_changes=(0.2, 10.0),
        safety_scale=0.95,
        min_step=1e-15,
        max_step=1e15,
    ):
        self.abstol = abstol
        self.reltol = reltol
        self.max_changes = max_changes
        self.safety_scale = safety_scale
        self.min_step = min_step
        self.max_step = max_step

    def __repr__(self):
        return f"{self.__class__.__name__}(abstol={self.abstol}, reltol={self.reltol})"

    @partial(jax.jit, static_argnums=0)
    def suggest(self, previous_dt, scaled_error_estimate, local_convergence_rate=None):
        if local_convergence_rate is None:
            raise ValueError("Please provide a local convergence rate.")

        small, large = self.max_changes

        ratio = 1.0 / scaled_error_estimate
        change = self.safety_scale * ratio ** (1.0 / local_convergence_rate)

        change = jnp.maximum(small, jnp.minimum(change, large))
        dt = change * previous_dt
        return dt

    @partial(jax.jit, static_argnums=0)
    def is_accepted(self, scaled_error_estimate):
        return scaled_error_estimate < 1

    @partial(jax.jit, static_argnums=0)
    def scale_error_estimate(self, unscaled_error_estimate, reference_state):
        if (
            not jnp.isscalar(unscaled_error_estimate.size)
        ) and unscaled_error_estimate.shape != reference_state.shape:
            raise ValueError(
                "Unscaled error estimate needs same shape as reference state."
            )
        tolerance = self.abstol + self.reltol * reference_state
        ratio = unscaled_error_estimate / tolerance
        dim = len(ratio) if ratio.ndim > 0 else 1
        return jnp.linalg.norm(ratio) / jnp.sqrt(dim)

    @partial(jax.jit, static_argnums=(0, 1, 5, 6))
    def first_dt(self, f, t0, tmax, y0, df, df_diagonal):
        return propose_first_dt(f, t0, y0)


@partial(jax.jit, static_argnums=0)
def propose_first_dt(f, t0, y0):
    norm_y0 = jnp.linalg.norm(y0)
    norm_dy0 = jnp.linalg.norm(f(t0, y0))
    return 0.01 * norm_y0 / norm_dy0
