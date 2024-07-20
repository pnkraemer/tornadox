"""Collect all modules into the tornadox.* namespace"""

import jax

from . import ek0, ek1, experimental, init, ivp, iwp, kalman, odefilter, rv, sqrt, step

jax.config.update("jax_enable_x64", True)


from ._version import version as __version__