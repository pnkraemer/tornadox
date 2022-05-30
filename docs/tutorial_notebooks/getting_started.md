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
from tornadox import ivp_examples
from tornadox import solve
from tornadox.solvers import ek1
from scipy.integrate import solve_ivp
```

Let's create an example ODE problem: non-stiff van der Pol.
You can write your own, but there are a couple of pre-baked recipes.

The recipes just return plain callable, tspans, etc..



```python
# Create an ODE problem
f, tspan, u0 = ivp_examples.vanderpol(stiffness_constant=1)
df = jax.jacfwd(f)
f, df = jax.jit(f), jax.jit(df)
print(f, tspan, u0)
```

Next, we choose a solver.
There are a couple of suggestions. For example, let us solve the ODE for the terminal value and use the Kronecker EK0 (the most efficient solver we have).

Solvers are tuples of an `init_fn` and a `perform_step_fn`, similar to how `optax` handles optimisers, and `blackjax` handles samplers. The `perform_step_fn` has error estimation and calibration baked into the implementation.

```python
solver = ek1.ek1_terminal_value(ode_dimension=2, num_derivatives=4)

print(solver)
```

Let's solve the ODE now.

```python
t, rv_terminal, _ = solve.solve_ivp_for_terminal_value(
    f=f, df=df, tspan=tspan, u0=u0, solver=solver, atol=1e-4, rtol=1e-4
)
m, c_sqrtm = rv_terminal

print()
print(m[0 :: (4 + 1)], t)
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

%timeit -n2 -r2 solve.solve_ivp_for_terminal_value(f=f, df=df, tspan=tspan, u0=u0, solver=solver, atol=1e-4, rtol=1e-4)
%timeit -n2 -r2 solve_ivp(f_not_autonomous, tspan, u0, method="RK45", atol=1e-4, rtol=1e-4, t_eval=(tspan[1],))
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

%timeit -n2 -r2 odeint(func=f_not_autonomous_swapped, y0=u0, t=jnp.array([tspan[0], tspan[1]]), atol=1e-4, rtol=1e-4)
```
