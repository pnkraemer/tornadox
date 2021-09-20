# tornadox
Lightweight, probabilistic ODE solvers. Fast like the wind. 🌪️ Powered by JAX.


## Usage
Use `tornadox` as follows.

```python
import jax.numpy as jnp
import jax.random
from tornadox import ek0, ek1, init, step, ivp, enkf

# Create a solver. Any of the following work. 
# The signatures of all solvers coincide.
solver1 = ek0.KroneckerEK0()
solver2 = ek0.ReferenceEK0(num_derivatives=6)
solver3 = ek1.ReferenceEK1(initialization=init.TaylorMode())
solver4 = ek1.DiagonalEK1(initialization=init.RungeKutta())
solver5 = ek1.ReferenceEK1(num_derivatives=5, steprule=step.AdaptiveSteps())

# These also solve ODEs, but use them at your own risk.
solver6 = ek1.TruncationEK1(num_derivatives=5, steprule=step.ConstantSteps(0.1))
solver7 = ek1.EarlyTruncationEK1(steprule=step.AdaptiveSteps(abstol=1e-4, reltol=1e-2))
solver8 = enkf.EnK1(prng_key=jax.random.PRNGKey(1), ensemble_size=100, initialization=init.CompiledRungeKutta(use_df=True)) 
solver9 = enkf.EnK1(prng_key=jax.random.PRNGKey(1), ensemble_size=100, steprule=step.AdaptiveSteps(abstol=1e-4, reltol=1e-2)) 


# Solve an IVP
vdp = ivp.vanderpol(t0=0., tmax=1., stiffness_constant=1.0)

for solver in [solver1, solver2, solver3, solver4, solver5, solver6, solver7, solver8, solver9]:
    
    # Full solve
    print(solver)
    solver.solve(vdp)
    solver.solve(vdp, stop_at=jnp.array([1.2, 1.3]))
    
    # Only solve for the final state
    solver.simulate_final_state(vdp)
    
    # Or go straight to the generator.
    for state, info in solver.solution_generator(vdp):
        pass
    print(info)
    
    print()
```
