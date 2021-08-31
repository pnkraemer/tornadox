"""ODE solver interface."""

import dataclasses
from abc import ABC, abstractmethod
from typing import Iterable, Union

import jax.numpy as jnp
import numpy as np

from tornadox import ek0, init, ivp, rv, step


@dataclasses.dataclass
class ODEFilterState:

    ivp: ivp.InitialValueProblem
    t: float
    y: Union[rv.MultivariateNormal, rv.MatrixNormal, rv.BatchedMultivariateNormal]
    error_estimate: jnp.ndarray
    reference_state: jnp.ndarray


@dataclasses.dataclass(frozen=False)
class ODESolution:
    t: Iterable[float]
    mean: Iterable[jnp.ndarray]
    cov_sqrtm: Iterable[jnp.ndarray]
    cov: Iterable[jnp.ndarray]


class ODEFilter(ABC):
    """Interface for filtering-based ODE solvers in ProbNum."""

    def __init__(self, *, steprule=None, num_derivatives=4, initialization=None):

        # Step-size selection
        self.steprule = steprule or step.AdaptiveSteps()

        # Number of derivatives
        self.num_derivatives = num_derivatives

        # IWP(nu) prior -- will be assembled in initialize()
        self.iwp = None

        # Initialization strategy
        self.init = initialization or init.TaylorMode()

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives={self.num_derivatives}, steprule={self.steprule}, initialization={self.init})"

    def solve(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        means = []
        covs = []
        cov_sqrtms = []
        times = []
        for state in solution_generator:
            times.append(state.t)
            means.append(state.y.mean)
            if isinstance(self, ek0.KroneckerEK0):
                cov_sqrtms.append(state.y.dense_cov_sqrtm())
                covs.append(state.y.dense_cov())
            else:
                cov_sqrtms.append(state.y.cov_sqrtm)
                covs.append(state.y.cov)

        return ODESolution(t=times, mean=means, cov_sqrtm=cov_sqrtms, cov=covs)

    def simulate_final_state(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        for state in solution_generator:
            pass
        return state

    def solution_generator(self, ivp, stop_at=None):
        """Generate ODE solver steps."""

        time_stopper = self._process_event_inputs(stop_at_locations=stop_at)
        state = self.initialize(ivp)
        info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_steps=0,
            num_attempted_steps=0,
        )
        yield state, info

        dt = self.steprule.first_dt(ivp)

        # Use state.ivp in case a callback modifies the IVP
        while state.t < state.ivp.tmax:
            if time_stopper is not None:
                dt = time_stopper.adjust_dt_to_time_stops(state.t, dt)

            state, dt, step_info = self.perform_full_step(state, dt)

            info["num_steps"] += 1
            info["num_f_evaluations"] += step_info["num_f_evaluations"]
            info["num_df_evaluations"] += step_info["num_df_evaluations"]
            info["num_df_diagonal_evaluations"] += step_info[
                "num_df_diagonal_evaluations"
            ]
            info["num_attempted_steps"] += step_info["num_attempted_steps"]
            yield state, info

    @staticmethod
    def _process_event_inputs(stop_at_locations):
        """Process callbacks and time-stamps into a format suitable for solve()."""

        if stop_at_locations is not None:
            time_stopper = _TimeStopper(stop_at_locations)
        else:
            time_stopper = None
        return time_stopper

    def perform_full_step(self, state, initial_dt):
        """Perform a full ODE solver step.

        This includes the acceptance/rejection decision as governed by error estimation
        and steprule.
        """
        dt = initial_dt
        step_is_sufficiently_small = False
        proposed_state = None
        step_info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_attempted_steps=0,
        )
        while not step_is_sufficiently_small:
            proposed_state, attempt_step_info = self.attempt_step(state, dt)

            step_info["num_attempted_steps"] += 1
            step_info["num_f_evaluations"] += (
                attempt_step_info["num_f_evaluations"]
                if "num_f_evaluations" in attempt_step_info
                else 0
            )
            step_info["num_df_evaluations"] += (
                attempt_step_info["num_df_evaluations"]
                if "num_df_evaluations" in attempt_step_info
                else 0
            )
            step_info["num_df_diagonal_evaluations"] += (
                attempt_step_info["num_df_diagonal_evaluations"]
                if "num_df_diagonal_evaluations" in attempt_step_info
                else 0
            )

            # Acceptance/Rejection due to the step-rule
            internal_norm = self.steprule.scale_error_estimate(
                unscaled_error_estimate=proposed_state.error_estimate,
                reference_state=proposed_state.reference_state,
            )
            step_is_sufficiently_small = self.steprule.is_accepted(internal_norm)
            suggested_dt = self.steprule.suggest(
                dt, internal_norm, local_convergence_rate=self.num_derivatives + 1
            )
            # Get a new step-size for the next step
            if step_is_sufficiently_small:
                dt = min(suggested_dt, state.ivp.tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, state.ivp.tmax - state.t)

        return proposed_state, dt, step_info

    @abstractmethod
    def initialize(self, ivp):
        raise NotImplementedError

    @abstractmethod
    def attempt_step(self, state, dt):
        raise NotImplementedError


class _TimeStopper:
    """Make the ODE solver stop at specified time-points."""

    def __init__(self, locations: Iterable):
        self._locations = iter(locations)
        self._next_location = next(self._locations)

    def adjust_dt_to_time_stops(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""

        if t >= self._next_location:
            try:
                self._next_location = next(self._locations)
            except StopIteration:
                self._next_location = np.inf

        if t + dt > self._next_location:
            dt = self._next_location - t
        return dt
