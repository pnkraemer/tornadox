# tornadox
Lightweight, probabilistic ODE solvers. Fast like the wind. 🌪️ Powered by JAX.


## Installation

Install `tornadox` via
```
pip install tornadox
```
Or get the most recent version from source:
```
pip install git+https://github.com/pnkraemer/tornadox.git
```


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
The efficient implementation of ODE filters is explained in the paper ([link](https://proceedings.mlr.press/v162/kramer22b.html))
```

@InProceedings{pmlr-v162-kramer22b,
  title = 	 {Probabilistic {ODE} Solutions in Millions of Dimensions},
  author =       {Kr{\"a}mer, Nicholas and Bosch, Nathanael and Schmidt, Jonathan and Hennig, Philipp},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {11634--11649},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/kramer22b/kramer22b.pdf},
  url = 	 {https://proceedings.mlr.press/v162/kramer22b.html}
}

```
Please consider citing it if you use this repository for your research.
