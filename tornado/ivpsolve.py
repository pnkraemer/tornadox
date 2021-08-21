import dataclasses
import functools
from typing import Dict, Iterable, Optional, Union

import jax.numpy as jnp

from tornado import ek0, ek1, ivp, odesolver, rv, step

# Will be extended in the dev process
_SOLVER_REGISTRY: Dict[str, odesolver.ODEFilter] = {
    "ek1_reference": ek1.ReferenceEK1,
    "ek1_diagonal": ek1.DiagonalEK1,
    "ek1_truncated": ek1.TruncatedEK1,
    "ek0_reference": ek0.ReferenceEK0,
    "ek0_kronecker": ek0.KroneckerEK0,
}


@dataclasses.dataclass(frozen=False)
class ODESolution:
    t: Iterable[float]
    mean: Iterable[jnp.ndarray]
    cov_sqrtm: Iterable[jnp.ndarray]
    cov: Iterable[jnp.ndarray]


def solve(
    ivp: ivp.InitialValueProblem,
    method: str = "ek1_ref",
    num_derivatives=5,
    adaptive: bool = True,
    dt: Optional[Union[float, step.StepRule]] = None,
    abstol: float = 1e-2,
    reltol: float = 1e-2,
    save_every_step=False,
):

    """Convenience function to solve IVPs.

    Parameters
    ----------
    ivp:
        Initial value problem.
    num_derivatives
        Number of derivatives of the integrated Wiener process prior.
    method : str, optional
        Which method is to be used.
    adaptive :
        Whether to use adaptive steps or not. Default is `True`.
    dt :
        Step size. If tolerances are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen automatically.
    abstol : float
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional.
    reltol : float
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional.
    save_every_step: bool
        Whether or not to save all results. If True, then no intermediate results are
        kept, save the last time point, in order to isolate the filtering itself,
        for timing. Optional. Per default, intermediate steps are discarded.

    Returns
    -------
    solution: ODESolution
        Solution of the ODE problem.
    solver: ODEFilter
        The solver object used to generate the solution.
        Through this object, projection matrices can be accessed.
    """

    # Create steprule
    if adaptive:
        if abstol is None or reltol is None:
            raise ValueError(
                "Please provide absolute and relative tolerance for adaptive steps."
            )
        first_dt = dt if dt is not None else step.propose_first_dt(ivp)
        steprule = step.AdaptiveSteps(first_dt=first_dt, abstol=abstol, reltol=reltol)
    else:
        steprule = step.ConstantSteps(dt)

    # Set up solve-algorithm
    try:
        solver = _SOLVER_REGISTRY[method](
            num_derivatives=num_derivatives,
            ode_dimension=ivp.dimension,
            steprule=steprule,
        )
    except KeyError:
        raise KeyError(
            f"Specified method {method} is unknown. "
            f"Known methods are {list(_SOLVER_REGISTRY.keys())}"
        )

    solution_generator = solver.solution_generator(ivp=ivp)

    if not save_every_step:
        for state in solution_generator:
            pass
        return state, solver

    # If not in benchmark (e.g. for testing/debugging), save and return results.
    res_means = []
    res_covs = []
    res_cov_sqrtms = []
    res_times = []
    for state in solution_generator:
        res_times.append(state.t)
        res_means.append(state.y.mean)
        if isinstance(solver, ek0.KroneckerEK0):
            res_cov_sqrtms.append(state.y.dense_cov_sqrtm())
            res_covs.append(state.y.dense_cov())
        else:
            res_cov_sqrtms.append(state.y.cov_sqrtm)
            res_covs.append(state.y.cov)

    return (
        ODESolution(
            t=res_times, mean=res_means, cov_sqrtm=res_cov_sqrtms, cov=res_covs
        ),
        solver,
    )
