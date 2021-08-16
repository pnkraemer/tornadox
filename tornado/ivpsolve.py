import dataclasses
from typing import Dict, Iterable, Optional, Union

import numpy as np

from tornado import ek1, ivp, odesolver, rv, step

# Will be extended in the dev process
_SOLVER_REGISTRY: Dict[str, odesolver.ODESolver] = {
    "ek1_diag": ek1.ReferenceEK1,
}


@dataclasses.dataclass
class ODEsolution:
    locations: Union[np.ndarray, Iterable[float]]
    y: Iterable[rv.MultivariateNormal]

    def mean(self):
        return [_y.mean for _y in self.y]

    def std(self):
        return [np.sqrt(np.diag(_y.cov)) for _y in self.y]

    def cov(self):
        return [_y.cov for _y in self.y]


def solve(
    ivp: ivp.InitialValueProblem,
    method: str = "ek1_diag",
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
    adaptive :
        Whether to use adaptive steps or not. Default is `True`.
    atol : float
        Absolute tolerance  of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    rtol : float
        Relative tolerance   of the adaptive step-size selection scheme.
        Optional. Default is ``1e-4``.
    dt :
        Step size. If atol and rtol are not specified, this step-size is used for a fixed-step ODE solver.
        If they are specified, this only affects the first step. Optional.
        Default is None, in which case the first step is chosen as prescribed by :meth:`propose_firststep`.


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
        solver = _SOLVER_REGISTRY["ek1_diag"](
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
    res_times = []
    for state in solver.solution_generator(ivp=ivp):
        res_times.append(state.t)
        res_states.append(state.y)

    return ODEsolution(locations=res_times, y=res_states), solver
