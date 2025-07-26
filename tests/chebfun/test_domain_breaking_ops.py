"""Unit-tests for Domain Breaking Operations in chebpy/core/chebfun.py"""

import pytest
import numpy as np

from chebpy import chebfun
from chebpy.core.settings import DefaultPreferences

from ..utilities import infnorm

# aliases
eps = DefaultPreferences.eps


def test_maximum_multipiece():
    x = chebfun("x", np.linspace(-2, 3, 11))
    y = chebfun(2, x.domain)
    g = (x**y).maximum(1.5)
    t = np.linspace(-2, 3, 2001)

    def f(x):
        return np.maximum(x**2, 1.5)

    assert infnorm(f(t) - g(t)) <= 1e1 * eps


def test_minimum_multipiece():
    x = chebfun("x", np.linspace(-2, 3, 11))
    y = chebfun(2, x.domain)
    g = (x**y).minimum(1.5)
    t = np.linspace(-2, 3, 2001)

    def f(x):
        return np.minimum(x**2, 1.5)

    assert infnorm(f(t) - g(t)) <= 1e1 * eps


# Define test parameters
domain_break_op_params = []
for domainBreakOp in (np.maximum, np.minimum):
    for n, args in enumerate([
        (lambda x: x, 0, [-1, 1], eps),
        (np.sin, np.cos, [-1, 1], eps),
        (np.cos, np.abs, [-1, 0, 1], eps),
    ]):
        f, g, dom, tol = args
        domain_break_op_params.append((domainBreakOp, f, g, dom, tol, n))


# Parametrized test for domain breaking operations
@pytest.mark.parametrize("domainBreakOp,f,g,dom,tol,n", domain_break_op_params)
def test_domain_break_op(domainBreakOp, f, g, dom, tol, n):
    xx = np.linspace(dom[0], dom[-1], 1001)
    ff = chebfun(f, dom)
    gg = chebfun(g, dom)

    # convert constant g to callable
    if isinstance(g, (int, float)):
        ffgg = domainBreakOp(f(xx), g)
    else:
        ffgg = domainBreakOp(f(xx), g(xx))

    fg = getattr(ff, domainBreakOp.__name__)(gg)

    vscl = max([ff.vscale, gg.vscale])
    hscl = max([ff.hscale, gg.hscale])
    lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])

    assert infnorm(fg(xx) - ffgg) <= vscl * hscl * lscl * tol
