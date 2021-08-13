"""Stepsize selection strategies."""

import jax.numpy as jnp
import abc

def propose_first_dt(ivp):
    norm_y0 = jnp.linalg.norm(ivp.y0)
    norm_dy0 = jnp.linalg.norm(ivp.f(ivp.t0, ivp.y0))
    return 0.01 * norm_y0 / norm_dy0


class StepRule(abc.ABC):
    """Step-size selection rules for ODE solvers."""

    def __init__(self, first_dt):
        self.first_dt = first_dt

    @abc.abstractmethod
    def suggest(
        self,  previous_dt, scaled_error_estimate, local_convergence_rate=None
    ):
        raise NotImplementedError

    @abc.abstractmethod
    def is_accepted(self, scaled_error_estimate):
        raise NotImplementedError

    def errorest_to_norm(
            self, unscaled_error_estimate, reference_state
    ):
        raise NotImplementedError


class ConstantSteps(StepRule):
    """Constant step-sizes."""

    def __init__(self, dt):
        self.dt = dt
        super().__init__(first_dt=dt)

    def suggest(
        self,  previous_dt, scaled_error_estimate, local_convergence_rate=None
    ):
        return self.dt

    def is_accepted(self, scaled_error_estimate):
        return True
    #
    # def errorest_to_norm(
    #         self, unscaled_error_estimate, reference_state
    # ):
    #     pass



class AdaptiveSteps(StepRule):

    def __init__(
        self,
        first_dt, abstol, reltol, max_changes = (0.2, 10.0), safety_scale= 0.95, min_step = 1e-15, max_step = 1e15
    ):
        super().__init__(first_dt=first_dt)
        self.abstol = abstol
        self.reltol = reltol
        self.max_changes = max_changes
        self.safety_scale = safety_scale
        self.min_step = min_step
        self.max_step = max_step

    def suggest(
            self, previous_dt, scaled_error_estimate, local_convergence_rate=None
    ):
        if local_convergence_rate is None:
            raise ValueError("Please provide a local convergence rate.")

        small, large = self.max_changes

        ratio = 1.0 / scaled_error_estimate
        change = self.safety_scale * ratio ** (1.0 / local_convergence_rate)

        # The below code should be doable in a single line?
        if change < small:
            dt = small * previous_dt
        elif large < change:
            dt = large * previous_dt
        else:
            dt = change * previous_dt

        if dt < self.min_step:
            raise ValueError("Step-size smaller than minimum step-size")
        if dt > self.max_step:
            raise ValueError("Step-size larger than maximum step-size")
        return dt

    def is_accepted(self, scaled_error_estimate):
        return scaled_error_estimate < 1

    def errorest_to_norm(
            self, unscaled_error_estimate, reference_state
    ):
        if unscaled_error_estimate.shape != reference_state.shape:
            raise ValueError("Unscaled error estimate needs same shape as reference state.")
        tolerance = self.abstol + self.reltol * reference_state
        ratio = unscaled_error_estimate / tolerance
        dim = len(ratio) if ratio.ndim > 0 else 1
        return jnp.linalg.norm(ratio) / jnp.sqrt(dim)
