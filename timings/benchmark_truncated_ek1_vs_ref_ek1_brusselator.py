"""Does truncation in the EK1 do something for high-ish dimensional problems?"""

import jax.numpy as jnp

import tornadox


def solve(ivp, solver):
    """Convenience access"""
    for idx, state in enumerate(solver.solution_generator(ivp=ivp)):
        pass
    return state, idx


def error(m1, m2):
    """Check discrepancy between solutions"""
    return jnp.linalg.norm((m1 - m2) / m1) / jnp.sqrt(m1.size)


# Set up the brusselator test problem
N = 20  # ode dimension will be d=2*N
bruss = tornadox.ivp.brusselator(N=N)

# Adaptive steps with medium/high accuracy
tolerance = 1e-5
first_dt = tornadox.step.propose_first_dt(ivp=bruss)
steps = tornadox.step.AdaptiveSteps(
    first_dt=first_dt, abstol=tolerance, reltol=tolerance
)

# Assemble both solvers
nu = 4
truncated_solver = tornadox.ek1.TruncationEK1(
    num_derivatives=nu, ode_dimension=bruss.dimension, steprule=steps
)
reference_solver = tornadox.ek1.ReferenceEK1(
    num_derivatives=nu, ode_dimension=bruss.dimension, steprule=steps
)

truncated_solution, num_steps_trunc = solve(ivp=bruss, solver=truncated_solver)
reference_solution, num_steps_ref = solve(ivp=bruss, solver=reference_solver)

print("Number of steps:")
print(f"\tTruncated: {num_steps_trunc}")
print(f"\tReference: {num_steps_ref}")


# Check the outputs are roughly equal
refmean = reference_solution.y.mean
truncmean = truncated_solution.y.mean.reshape(refmean.shape, order="F")
print(f"Discrepancy: {error(truncmean, refmean)}")
assert jnp.allclose(refmean, truncmean, rtol=1e-2, atol=1e-2)


# %timeit solve(ivp=bruss, solver=truncated_solver)
# --> 7.94 s ± 970 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %timeit solve(ivp=bruss, solver=reference_solver)
# --> 21.5 s ± 1.8 s per loop (mean ± std. dev. of 7 runs, 1 loop each)
