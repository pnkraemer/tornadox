---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Getting started

How can I use the code in `odefilter` to solve ODEs?


```python
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from pnx.example_zoo import ivp as ivp_examples
from pnx.odefilter import solve
from pnx.odefilter.solvers.first_order import ek0
from scipy.integrate import solve_ivp
```

Let's create an example ODE problem: non-stiff van der Pol.
You can write your own, but there are a couple of pre-baked recipes.

The recipes just return plain callable, tspans, etc..



```python
# Create an ODE problem
f, tspan, u0 = ivp_examples.vanderpol(stiffness_constant=1)

print(f, tspan, u0)
```

Next, we choose a solver.
There are a couple of suggestions. For example, let us solve the ODE for the terminal value and use the Kronecker EK0 (the most efficient solver we have).

Solvers are tuples of an `init_fn` and a `perform_step_fn`, similar to how `optax` handles optimisers, and `blackjax` handles samplers. The `perform_step_fn` has error estimation and calibration baked into the implementation.

```python
ek0_solver = ek0.ek0_kronecker_terminal_value(num_derivatives=4)

print(ek0)
```

Let's solve the ODE now.

```python
(rv_terminal, t), _ = solve.solve_ivp_for_terminal_value(
    f=f, df=None, tspan=tspan, u0=u0, solver=ek0_solver, atol=1e-4, rtol=1e-4
)
m, c_sqrtm = rv_terminal

print()
print(m[0, :], t)
```

The code is written in pure Jax, and we try to make it as efficient as possible.
Compare to scipy's solver:

```python
@jax.jit
def f_not_autonomous(_, y):
    return f(y)


# precompile (at least the ODE function)
solve_ivp(
    f_not_autonomous, tspan, u0, method="RK45", atol=1e-4, rtol=1e-4, t_eval=(tspan[1],)
)

%timeit solve.solve_ivp_for_terminal_value(f=f, df=None, tspan=tspan, u0=u0, solver=ek0_solver, atol=1e-4, rtol=1e-4)
%timeit solve_ivp(f_not_autonomous, tspan, u0, method="RK45", atol=1e-4, rtol=1e-4, t_eval=(tspan[1],))
```

It is quite comparable in performance to Jax's ODE solver in performance.

```python
@jax.jit
def f_not_autonomous_swapped(y, _):
    return f(y)


# precompile
odeint(
    func=f_not_autonomous_swapped,
    y0=u0,
    t=jnp.array([tspan[0], tspan[1]]),
    atol=1e-4,
    rtol=1e-4,
)

%timeit odeint(func=f_not_autonomous_swapped, y0=u0, t=jnp.array([tspan[0], tspan[1]]), atol=1e-4, rtol=1e-4)
```
