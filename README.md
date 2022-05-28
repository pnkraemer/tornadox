# tornadox
Lightweight, probabilistic ODE solvers. Fast like the wind. 🌪️ Powered by JAX.


## Installation
`tornadox` depends on XLA via JAX. 
But since the JAX installation depends on you personal choices (CPU/GPU/TPU, etc.),
`tornadox` does not list a JAX dependency.

If you want the CPU version of JAX, install it via
```
pip install jax[cpu]
```
and then  follow up with installing `tornadox` via
```
pip install tornadox
```
Or get the most recent version from source:
```
pip install git+https://github.com/pnkraemer/tornadox.git
```
A word of caution: at the moment, the tests are executed on CPU only.
Things are not guaranteed to work on GPU and TPU. 

## Usage
Use `tornadox` as follows.

```python
import jax.numpy as jnp
from tornadox import ek0, ek1, init, step, ivp

# Create a solver. Any of the following work. 
# The signatures of all solvers coincide.
solver1 = ek0.KroneckerEK0()
solver2 = ek0.ReferenceEK0(num_derivatives=6)
solver3 = ek1.ReferenceEK1(initialization=init.TaylorMode())
solver4 = ek1.DiagonalEK1(initialization=init.RungeKutta())
solver5 = ek1.ReferenceEK1(num_derivatives=5, steprule=step.AdaptiveSteps())

# Solve an IVP
vdp = ivp.vanderpol(t0=0., tmax=1., stiffness_constant=1.0)

for solver in [solver1, solver2, solver3, solver4, solver5]:
    
    # Full solve
    print(solver)
    solver.solve(vdp)
    solver.solve(vdp, stop_at=jnp.array([1.2, 1.3]))
    
    # Only solve for the final state
    solver.simulate_final_state(vdp)
    
    # Or go straight to the generator
    for state, info in solver.solution_generator(vdp):
        pass
    print(info)
    
    print()
```


## Citation
The efficient implementation of ODE filters is explained in the paper ([link](https://arxiv.org/abs/2110.11812))
```
@misc{krämer2021probabilistic,
      title={Probabilistic ODE Solutions in Millions of Dimensions}, 
      author={Nicholas Krämer and Nathanael Bosch and Jonathan Schmidt and Philipp Hennig},
      year={2021},
      eprint={2110.11812},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
Please consider citing it if you use this repository for your research.
