# tornado
Lightweight, probabilistic ODE solvers. Fast like the wind. 🌪️


## Usage

```python
import jax.numpy as jnp

from tornado import ek0, ek1, init, step, ivp

# Create a solver. Any of the following work. 
# The signatures of all solvers coincide.
solver1 = ek0.KroneckerEK0()
solver2 = ek0.ReferenceEK0(num_derivatives=6)
solver3 = ek1.ReferenceEK1(initialization=init.TaylorMode())
solver4 = ek1.DiagonalEK1(initialization=init.RungeKutta())
solver5 = ek1.TruncationEK1(num_derivatives=5, steprule=step.ConstantSteps(0.1))
solver6 = ek1.ReferenceEK1(num_derivatives=5, steprule=step.AdaptiveSteps())
solver7 = ek1.EarlyTruncationEK1(steprule=step.AdaptiveSteps(abstol=1e-4, reltol=1e-2))

# Solve an IVP
ivp = ivp.vanderpol(t0=0., tmax=1.)

for solver in [solver1, solver2, solver3, solver4, solver5, solver6, solver7]:
    
    # Full solve
    print(solver)
    solver.solve(ivp)
    solver.solve(ivp, stop_at=jnp.array([1.2, 1.3]))
    
    # Only solve for the final state
    solver.simulate_final_state(ivp)
    
    # Go straight to the generator
    for state in solver.solution_generator(ivp):
        print(state.t, state.y.mean)

    print()
```