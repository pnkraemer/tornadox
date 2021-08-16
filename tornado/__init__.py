"""Collect all modules into the tornado.* namespace"""

from jax.config import config

from . import (
    ek1,
    ivp,
    ivpsolve,
    iwp,
    linops,
    odesolver,
    rv,
    sqrt,
    step,
    taylor_mode,
    ek0,
)

config.update("jax_enable_x64", True)
