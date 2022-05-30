"""Solvers."""

from functools import partial

import jax


@partial(jax.jit, static_argnames=("f", "df", "solver"))
def solve_ivp_for_terminal_value(*, f, df, tspan, u0, solver, **solver_kwargs):
    r"""Solve an initial value problem and only store the terminal value.

    The solver is a triple of functions:
    init_fn, perform_step_fn, extract_qoi_fn = solver

    This is similar to how Jax recommends to implement optimisers
    ``https://jax.readthedocs.io/en/latest/jax.example_libraries.optimizers.html``
    which is done by e.g. optax. blackjax also follows a similar pattern.
    (And so does scipy.integrate.OdeSolver, coincidentally.)

    Signature:

    ``init_fn(*, f, df, u0, tspan,  **solver_kwargs) -> (t, MyCustomState)``

    ``perform_step_fn(t, custom_state, *, f, df, t1, u0, **solver_kwargs) -> (t, MyCustomState)``

    ``extract_qoi_fn(t, custom_state) -> Any``


    CustomState can really be anything and is purely internal,
    i.e., it is unique to each solver and a user never gets to see it.
    (It must be a pytree, though.)
    Internally, we usually include `dt_proposed`, `u`, etc.
    in a single big (named)tuple.

    solver_kwargs are atol, rtol, dt_min, etc.,
    which are again quite unique to each solver.
    """  # noqa: E501 # todo: improve docstring :)

    init_fn, perform_step_fn, extract_qoi_fn = solver

    # Call partial() internally,
    # because we don't want to recompile for different tolerances.
    init_fn = partial(init_fn, **solver_kwargs)
    perform_step_fn = partial(perform_step_fn, **solver_kwargs)

    # Initialise and loop over interval
    # todo: even f, df, and u0 could be unique to each solver
    #  and swallowed up by some kwargs.
    #  Only tspan is needed on this level. Should we do this?
    t0, state0 = init_fn(f=f, df=df, tspan=tspan, u0=u0)
    t, state = _solve_ivp_on_interval(
        f=f,
        df=df,
        t1=tspan[1],
        t0=t0,
        state0=state0,
        perform_step_fn=perform_step_fn,
    )
    return extract_qoi_fn(t, state)


@partial(jax.jit, static_argnames=("f", "df", "solver"))
def solve_ivp_saveat(*, f, df, u0, solver, saveat, **solver_kwargs):
    # assume t0 and tmax are the "boundary" elements in saveat.
    # As such, `saveat` makes `tspan` obsolete
    # Also assume it is sorted.
    # If not, anything could happen.

    # The signatures of those functions are the same as in solve_ivp_terminal_value
    # with the exception of the additional reset_state_at_checkpoint_fn().
    # It maps (t,state) to (t, state), and replaces some internals accordingly.
    # There is probably a cleaner way of implementing this, but for now it works.
    init_fn, perform_step_fn, reset_state_at_checkpoint_fn, extract_qoi_fn = solver

    # Call partial() internally,
    # because we don't want to recompile for different tolerances.
    init_fn = partial(init_fn, **solver_kwargs)
    perform_step_fn = partial(perform_step_fn, **solver_kwargs)

    @jax.jit
    def solve_for_next_saveat(s, t1):
        t0, state0 = s

        # Reset and solve on the next interval.
        t0, state0 = reset_state_at_checkpoint_fn(t0, state0)
        t, state = _solve_ivp_on_interval(
            f=f,
            df=df,
            t0=t0,
            t1=t1,
            state0=state0,
            perform_step_fn=perform_step_fn,
        )
        return (t, state), extract_qoi_fn(t, state)

    init_val = init_fn(f=f, df=df, tspan=(saveat[0], saveat[-1]), u0=u0)
    state_terminal, state_qoi = jax.lax.scan(
        f=solve_for_next_saveat,
        init=init_val,
        xs=saveat[1:],
        reverse=False,
    )

    # Return the initial value along with the rest, because
    # it contains components of the filtering solution
    return extract_qoi_fn(*state_terminal), state_qoi, extract_qoi_fn(*init_val)


@partial(jax.jit, static_argnames=("f", "df", "perform_step_fn"))
def _solve_ivp_on_interval(*, f, df, t1, t0, state0, perform_step_fn):
    """Solve an IVP adaptively on the interval (t0, t1)."""

    @jax.jit
    def cond_fun(s):
        t, _ = s
        return t < t1

    @jax.jit
    def body_fun(s):
        t, state = s
        return perform_step_fn(t, state, f=f, df=df, t1=t1)

    return jax.lax.while_loop(
        cond_fun=cond_fun,
        body_fun=body_fun,
        init_val=(t0, state0),
    )
