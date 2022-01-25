"""ODE solver interface."""

import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import Dict, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from tornadox import ek0, init, step


class ODEFilterState(
    namedtuple("_ODEFilterState", "t y error_estimate reference_state")
):
    pass


@dataclasses.dataclass(frozen=False)
class ODESolution:
    t: jnp.ndarray
    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray
    info: Dict


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
        cov_sqrtms = []
        times = []
        info = dict()

        for state, info in solution_generator:
            times.append(state.t)
            means.append(state.y.mean)
            if isinstance(self, ek0.KroneckerEK0):
                cov_sqrtms.append(state.y.dense_cov_sqrtm())
            else:
                cov_sqrtms.append(state.y.cov_sqrtm)

        return ODESolution(
            t=jnp.stack(times),
            mean=jnp.stack(means),
            cov_sqrtm=jnp.stack(cov_sqrtms),
            info=info,
        )

    def simulate_final_state(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        state, info = None, None
        for state, info in solution_generator:
            pass
        return state, info

    def solution_generator(
        self,
        ivp,
        stop_at=None,
        progressbar=False,
        compile_step=True,
        compile_init=False,
    ):
        """Generate ODE solver steps."""

        # Choose the compiled or non-compiled perform_full_step implementation.
        # For small problems (d << 1000), compiling accelerates things drastically;
        # for LARGE problems (d >> 1000), it seems to slow things
        # down---at least, at the moment.
        #
        # This is only an intermediate functionality:
        # I plan on replacing the "compile_step" flag with a new solve() interface soon,
        # that is, as soon as the WHOLE solve can be compiled. (This takes a few more changes.)
        choose_perform_step = {
            True: self.perform_full_step_compiled,
            False: self.perform_full_step,
        }
        perform_full_step = choose_perform_step[compile_step]
        if compile_init:
            initialize = jax.jit(self.initialize, static_argnums=(0, 1, 5, 6))
        else:
            initialize = self.initialize

        time_stopper = self._process_event_inputs(stop_at_locations=stop_at)
        state = initialize(*ivp)
        info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_steps=0,
            num_attempted_steps=0,
        )
        yield state, info

        dt = self.steprule.first_dt(*ivp)

        progressbar_steps = 100
        progressbar_update_threshold = progressbar_update_increment = (
            ivp.tmax / progressbar_steps
        )
        pbar = tqdm(total=progressbar_steps) if progressbar else None
        while state.t < ivp.tmax:

            if pbar is not None:
                while state.t + dt >= progressbar_update_threshold:
                    pbar.update()
                    progressbar_update_threshold += progressbar_update_increment
                pbar.set_description(f"t={state.t:.4f}, dt={dt:.2E}")

            if time_stopper is not None:
                dt = time_stopper.adjust_dt_to_time_stops(state.t, dt)

            state, dt, step_info = perform_full_step(state, dt, *ivp)

            # Todo: the following safety net has been removed for jitting reasons.
            # Todo: If we run into issues here, we have to add something back.
            # (The code is left for doc purposes)
            #
            # if dt < self.steprule.min_step:
            #     raise ValueError("Step-size smaller than minimum step-size")
            # if dt > self.steprule.max_step:
            #     raise ValueError("Step-size larger than maximum step-size")

            info["num_steps"] += 1
            info["num_f_evaluations"] += step_info["num_f_evaluations"]
            info["num_df_evaluations"] += step_info["num_df_evaluations"]
            info["num_df_diagonal_evaluations"] += step_info[
                "num_df_diagonal_evaluations"
            ]
            info["num_attempted_steps"] += step_info["num_attempted_steps"]
            yield state, info

        if pbar is not None:
            pbar.update()
            pbar.close()

    @staticmethod
    def _process_event_inputs(stop_at_locations):
        """Process callbacks and time-stamps into a format suitable for solve()."""

        if stop_at_locations is not None:
            time_stopper = _TimeStopper(stop_at_locations)
        else:
            time_stopper = None
        return time_stopper

    def perform_full_step(self, state, initial_dt, f, t0, tmax, y0, df, df_diagonal):
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

            proposed_state, attempt_step_info = self.attempt_step(
                state, dt, f, t0, tmax, y0, df, df_diagonal
            )

            # Gather some stats
            step_info["num_attempted_steps"] += 1
            if "num_f_evaluations" in attempt_step_info:
                nfevals = attempt_step_info["num_f_evaluations"]
                step_info["num_f_evaluations"] += nfevals
            if "num_df_evaluations" in attempt_step_info:
                ndfevals = attempt_step_info["num_df_evaluations"]
                step_info["num_df_evaluations"] += ndfevals
            if "num_df_diagonal_evaluations" in attempt_step_info:
                ndfevals_diag = attempt_step_info["num_df_diagonal_evaluations"]
                step_info["num_df_diagonal_evaluations"] += ndfevals_diag

            # Acceptance/Rejection due to the step-rule
            internal_norm = self.steprule.scale_error_estimate(
                unscaled_error_estimate=dt * proposed_state.error_estimate
                if proposed_state.error_estimate is not None
                else None,
                reference_state=proposed_state.reference_state,
            )
            step_is_sufficiently_small = self.steprule.is_accepted(internal_norm)
            suggested_dt = self.steprule.suggest(
                dt, internal_norm, local_convergence_rate=self.num_derivatives + 1
            )
            # Get a new step-size for the next step
            if step_is_sufficiently_small:
                dt = min(suggested_dt, tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, tmax - state.t)

            assert dt >= 0, f"Invalid step size: dt={dt}"

        return proposed_state, dt, step_info

    @partial(jax.jit, static_argnums=(0, 3, 7, 8))
    def perform_full_step_compiled(
        self, state, initial_dt, f, t0, tmax, y0, df, df_diagonal
    ):
        """Perform a full ODE solver step, but using jax control flow.

        This includes the acceptance/rejection decision as governed by error estimation
        and steprule.
        """

        # Implement the iteration of refining the steps until a step is small enough
        # via jax.lax.while_loop. It requires a value, a body() function and a condition() function.
        # Below we define those, and then carry out the loop.

        # Initialise the first instance of the data structure
        step_info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_attempted_steps=0,
        )
        init_val = ODEFilter.CarryingValue(
            i=0,
            dt=initial_dt,
            step_is_sufficiently_small=False,
            proposed_state=state,
            step_info=step_info,
            state=state,
            t0=t0,
            tmax=tmax,
            y0=y0,
        )

        # Fix all the non-jittable inputs (i.e. all the callables) via partial()
        full_step_body_fixed_inputs = partial(
            self.perform_full_step_body_fun, f=f, df=df, df_diagonal=df_diagonal
        )
        body_fun = jax.jit(full_step_body_fixed_inputs)
        cond_fun = jax.jit(ODEFilter.perform_full_step_cond_fun)

        # Do the actual loop
        final_value = jax.lax.while_loop(cond_fun, body_fun, init_val)

        # Return only the interesting quantities
        return final_value.proposed_state, final_value.dt, final_value.step_info

    # Define a reasonable data structure that is a valid jax type: (named)tuple
    # and that carries all information that we need to perform the next step attempt.
    CarryingValue = namedtuple(
        "CarryingValue",
        "i dt step_is_sufficiently_small proposed_state step_info state t0 tmax y0",
    )

    @staticmethod
    def perform_full_step_cond_fun(value):
        # Condition for whether to continue the while loop or not.
        # Check that the state was not too large
        #
        # We could add a maximum number of step attempts here
        return jnp.logical_not(value.step_is_sufficiently_small)

    # Not jitted... will be jitted later
    def perform_full_step_body_fun(self, value, f, df, df_diagonal):
        """The body of the while loop.

        Keep attempting steps until the error estimate suggests a sufficiently small step.
        """
        # Extract relevant states from current input
        # Ignore the step_is_sufficiently_small flag and the proposed_state, they are only in the tuple
        # because they are returned at the end of the loop.
        i, dt, _, _, step_info, state, t0, tmax, y0 = value

        # Attempt a step
        proposed_state, attempt_step_info = self.attempt_step(
            state, dt, f, t0, tmax, y0, df, df_diagonal
        )

        # Gather some stats
        step_info["num_attempted_steps"] += 1
        if "num_f_evaluations" in attempt_step_info:
            nfevals = attempt_step_info["num_f_evaluations"]
            step_info["num_f_evaluations"] += nfevals
        if "num_df_evaluations" in attempt_step_info:
            ndfevals = attempt_step_info["num_df_evaluations"]
            step_info["num_df_evaluations"] += ndfevals
        if "num_df_diagonal_evaluations" in attempt_step_info:
            ndfevals_diag = attempt_step_info["num_df_diagonal_evaluations"]
            step_info["num_df_diagonal_evaluations"] += ndfevals_diag

        # Acceptance/Rejection due to the step-rule
        internal_norm = self.steprule.scale_error_estimate(
            unscaled_error_estimate=dt * proposed_state.error_estimate
            if proposed_state.error_estimate is not None
            else None,
            reference_state=proposed_state.reference_state,
        )
        step_is_sufficiently_small = self.steprule.is_accepted(internal_norm)
        suggested_dt = self.steprule.suggest(
            dt, internal_norm, local_convergence_rate=self.num_derivatives + 1
        )

        # Get a new step-size for the next step
        # Roughly equivalent to: if step was good, pick either the proposed step or tmax-t.
        true_fun = lambda x: jnp.minimum(x[0], x[1] - x[2])
        false_fun = lambda x: jnp.minimum(x[0], x[1] - x[3])
        operand = suggested_dt, tmax, proposed_state.t, state.t
        dt = jax.lax.cond(step_is_sufficiently_small, true_fun, false_fun, operand)

        # Return all the important information to continue the loop
        return ODEFilter.CarryingValue(
            i + 1,
            dt,
            step_is_sufficiently_small,
            proposed_state,
            step_info,
            state,
            t0,
            tmax,
            y0,
        )

    @abstractmethod
    def initialize(self, f, t0, tmax, y0, df, df_diagonal):
        raise NotImplementedError

    @abstractmethod
    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):
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
