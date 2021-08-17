import dataclasses
import functools
from typing import Dict, Iterable, Optional, Union

import jax.numpy as jnp

from tornado import ek1, ivp, odesolver, rv, step

# Will be extended in the dev process
_SOLVER_REGISTRY: Dict[str, odesolver.ODEFilter] = {
    "ek1_ref": ek1.ReferenceEK1,
    "ek1_diag": ek1.DiagonalEK1,
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
    solver_order=5,
    adaptive: bool = True,
    dt: Optional[Union[float, step.StepRule]] = None,
    abstol: float = 1e-2,
    reltol: float = 1e-2,
    on_the_fly=True,
):

    """Convenience function to solve IVPs.

    Parameters
    ----------
    ivp:
        Initial value problem.
    solver_order
        Order of the solver. This amounts to choosing the number of derivatives of an integrated Wiener process prior.
        For too high orders, process noise covariance matrices become singular.
        For integrated Wiener processes, this maximum seems to be ``num_derivatives=11`` (using standard ``float64``).
        It is possible that higher orders may work for you.
        The type of prior relates to prior assumptions about the derivative of the solution.
        The higher the order of the solver, the faster the convergence, but also, the higher-dimensional (and thus the costlier) the state space.
    method : str, optional
        Which method is to be used.
    adaptive :
        Whether to use adaptive steps or not. Default is `True`.
    dt :
        Step size. If atol and rtol are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen as prescribed by :meth:`propose_firststep`.
    abstol : float
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    reltol : float
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    benchmark_mode: bool
        Whether or not to save the results. If True, then no intermediate results are
        kept, save the last time point, in order to isolate the filtering itself,
        for timing. Defaults to True.

    Returns
    -------
    solution: ODESolution
        Solution of the ODE problem.
        It contains fields:
        t :
            Mesh used by the solver to compute the solution.
        y :
            Discrete-time solution at times :math:`t_1, ..., t_N`,
            as a list of random variables.
    solver: ODEFilter
        The solver object used to generate the solution.
        Via this object, projection matrices can be accessed.
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
            num_derivatives=solver_order,
            ode_dimension=ivp.dimension,
            steprule=steprule,
        )
    except KeyError:
        raise KeyError(
            f"Specified method {method} is unknown. "
            f"Known methods are {list(_SOLVER_REGISTRY.keys())}"
        )

    solution_generator = solver.solution_generator(ivp=ivp)

    if on_the_fly:
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
        res_cov_sqrtms.append(state.y.cov_sqrtm)
        res_covs.append(state.y.cov)

    return (
        ODESolution(
            t=res_times, mean=res_means, cov_sqrtm=res_cov_sqrtms, cov=res_covs
        ),
        solver,
    )
