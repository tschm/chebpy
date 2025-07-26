"""Unit-tests for Plotting functionality in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.plotting import import_plt

from ..utilities import joukowsky

# aliases
sin = np.sin
cos = np.cos
exp = np.exp

plt = import_plt()


@pytest.fixture
def plotting_functions():
    def f(x):
        return sin(4 * x) + exp(cos(14 * x)) - 1.4

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    f1 = Chebfun.initfun_adaptive(f, [-1, 1])
    f2 = Chebfun.initfun_adaptive(f, [-3, 0, 1])
    f3 = Chebfun.initfun_adaptive(f, [-2, -0.3, 1.2])
    f4 = Chebfun.initfun_adaptive(u, [-1, 1])

    return f1, f2, f3, f4


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot(plotting_functions):
    f1, f2, f3, _ = plotting_functions
    for fun in [f1, f2, f3]:
        fig, ax = plt.subplots()
        fun.plot(ax=ax)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_complex(plotting_functions):
    _, _, _, f4 = plotting_functions
    fig, ax = plt.subplots()
    # plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        (np.exp(1j * 0.5 * np.pi) * joukowsky(rho * f4)).plot(ax=ax)


@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plotcoeffs(plotting_functions):
    f1, f2, f3, _ = plotting_functions
    for fun in [f1, f2, f3]:
        fig, ax = plt.subplots()
        fun.plotcoeffs(ax=ax)
