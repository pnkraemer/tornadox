import dataclasses
import functools
from typing import Dict, Iterable, Optional, Union

import jax.numpy as jnp

from tornado import ek1, ivp, odesolver, rv, step

# Will be extended in the dev process
_SOLVER_REGISTRY: Dict[str, odesolver.ODEFilter] = {
    "ek1_ref": ek1.ReferenceEK1,
}


@dataclasses.dataclass(frozen=False)
class ODESolution:
    t: Union[jnp.ndarray, Iterable[float]]
    y: Iterable[rv.MultivariateNormal]
    means: Optional[Iterable[jnp.ndarray]] = None
    covs_cholesky: Optional[Iterable[jnp.ndarray]] = None

    @property
    def mean(self):
        if self.means is None:
            self.means = [_y.mean for _y in self.y]

        return self.means

    @property
    def cov_cholesky(self):
        if self.covs_cholesky is None:
            self.covs_cholesky = [_y.cov_cholesky for _y in self.y]

        return self.covs_cholesky

    @functools.cached_property
    def cov(self):
        return [_L @ _L.T for _L in self.cov_cholesky]


def solve(
    ivp: ivp.InitialValueProblem,
    method: str = "ek1_ref",
    solver_order=2,
    adaptive: bool = True,
    dt: Optional[Union[float, step.StepRule]] = None,
    abstol: float = 1e-2,
    reltol: float = 1e-2,
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
        firststep = dt if dt is not None else step.propose_firststep(ivp)
        steprule = step.AdaptiveSteps(firststep=firststep, abstol=abstol, reltol=reltol)
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

    res_states = []
    res_means = []
    res_cov_chols = []
    res_times = []
    for state in solver.solution_generator(ivp=ivp):
        res_times.append(state.t)
        res_states.append(state.y)
        res_means.append(state.y.mean)
        res_cov_chols.append(state.y.cov_cholesky)

    return (
        ODESolution(
            t=res_times, y=res_states, means=res_means, covs_cholesky=res_cov_chols
        ),
        solver,
    )
