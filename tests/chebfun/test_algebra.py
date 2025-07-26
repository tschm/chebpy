"""Unit-tests for chebpy/core/chebfun.py - Algebra class"""

import itertools
import operator
import pytest

import numpy as np

from chebpy import chebfun
from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from ..utilities import infnorm, testfunctions

# in Python 3, the operator module does not have a 'div' method
binops = [operator.add, operator.mul, operator.sub, operator.truediv]
try:
    # in Python 2 we need to test div separately
    binops.append(operator.div)
    div_binops = (operator.div, operator.truediv)
except AttributeError:
    # Python 3
    div_binops = (operator.truediv,)

# aliases
eps = DefaultPreferences.eps

# domain, test_tolerance
chebfun_testdomains = [
    ([-1, 1], 2 * eps),
    ([-2, 1], eps),
    ([-1, 2], eps),
    ([-5, 9], 35 * eps),
]

# Define powtestfuns and powtestdomains
powtestfuns = (
    [(np.exp, "exp"), (np.sin, "sin")],
    [(np.exp, "exp"), (lambda x: 2 - x, "linear")],
    [(lambda x: 2 - x, "linear"), (np.exp, "exp")],
)

powtestdomains = [
    ([-0.5, 0.9], eps),
    ([-1.2, 1.3], eps),
    ([-2.2, -1.9], eps),
    ([0.4, 1.3], eps),
]


@pytest.fixture
def algebra_functions():
    emptyfun = Chebfun.initempty()
    yy = np.linspace(-1, 1, 2000)
    return emptyfun, yy

# check  +(empty Chebfun) = (empty Chebfun)
def test__pos__empty(algebra_functions):
    emptyfun, _ = algebra_functions
    assert (+emptyfun).isempty


# check -(empty Chebfun) = (empty Chebfun)
def test__neg__empty(algebra_functions):
    emptyfun, _ = algebra_functions
    assert (-emptyfun).isempty


# check (empty Chebfun) + (Chebfun) = (empty Chebfun)
#   and (Chebfun) + (empty Chebfun) = (empty Chebfun)
def test__add__radd__empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (emptyfun + ff).isempty
            assert (ff + emptyfun).isempty


# check the output of (constant + Chebfun)
#                 and (Chebfun + constant)
def test__add__radd__constant():
    for f, _, _ in testfunctions:
        for c in (-1, 1, 10, -1e5):

            def g(x):
                return c + f(x)

            for dom, _ in chebfun_testdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg1 = c + ff
                gg2 = ff + c
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg1(xx)) <= tol
                assert infnorm(g(xx) - gg2(xx)) <= tol

# check (empty Chebfun) - (Chebfun) = (empty Chebfun)
#   and (Chebfun) - (empty Chebfun) = (empty Chebfun)
def test__sub__rsub__empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (emptyfun - ff).isempty
            assert (ff - emptyfun).isempty


# check the output of (constant - Chebfun)
#                 and (Chebfun - constant)
def test__sub__rsub__constant():
    for f, _, _ in testfunctions:
        for c in (-1, 1, 10, -1e5):

            def g(x):
                return c - f(x)

            for dom, _ in chebfun_testdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg1 = c - ff
                gg2 = ff - c
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg1(xx)) <= tol
                assert infnorm(-g(xx) - gg2(xx)) <= tol

# check (empty Chebfun) * (Chebfun) = (empty Chebfun)
#   and (Chebfun) * (empty Chebfun) = (empty Chebfun)
def test__mul__rmul__empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (emptyfun * ff).isempty
            assert (ff * emptyfun).isempty


# check the output of (constant * Chebfun)
#                 and (Chebfun * constant)
def test__mul__rmul__constant():
    for f, _, _ in testfunctions:
        for c in (-1, 1, 10, -1e5):

            def g(x):
                return c * f(x)

            for dom, _ in chebfun_testdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg1 = c * ff
                gg2 = ff * c
                vscl = ff.vscale
                hscl = ff.hscale
                lscl = max([fun.size for fun in ff])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg1(xx)) <= tol
                assert infnorm(g(xx) - gg2(xx)) <= tol

# check (empty Chebfun) / (Chebfun) = (empty Chebfun)
#   and (Chebfun) / (empty Chebfun) = (empty Chebfun)
def test_truediv_empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (emptyfun / ff).isempty
            assert (ff / emptyfun).isempty


# check the output of (constant / Chebfun)
#                 and (Chebfun / constant)
def test_truediv_constant():
    for f, _, hasRoots in testfunctions:
        for c in (-1, 1, 10, -1e5):

            def g(x):
                return f(x) / c

            def h(x):
                return c / f(x)

            for dom, _ in chebfun_testdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg = ff / c
                vscl = gg.vscale
                hscl = gg.hscale
                lscl = max([fun.size for fun in gg])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg(xx)) <= tol
                # don't do the following test for functions with roots
                if not hasRoots:
                    hh = c / ff
                    vscl = hh.vscale
                    hscl = hh.hscale
                    lscl = max([fun.size for fun in hh])
                    tol = 2 * abs(c) * vscl * hscl * lscl * eps
                    assert infnorm(h(xx) - hh(xx)) <= tol

# check (empty Chebfun) ** (Chebfun) = (empty Chebfun)
def test_pow_empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (emptyfun**ff).isempty


# check (Chebfun) ** (empty Chebfun) = (empty Chebfun)
def test_rpow_empty(algebra_functions):
    emptyfun, _ = algebra_functions
    for f, _, _ in testfunctions:
        for dom, _ in chebfun_testdomains:
            a, b = dom
            ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))
            assert (ff**emptyfun).isempty


# check the output of (Chebfun) ** (constant)
def test_pow_constant():
    for (_, _), (f, _) in powtestfuns:
        for c in (1, 2, 3):

            def g(x):
                return f(x) ** c

            for dom, _ in powtestdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg = ff**c
                vscl = gg.vscale
                hscl = gg.hscale
                lscl = max([fun.size for fun in gg])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg(xx)) <= tol


# check the output of (constant) ** (Chebfun)
def test_rpow_constant():
    for (_, _), (f, _) in powtestfuns:
        for c in (1, 2, 3):

            def g(x):
                return c ** f(x)

            for dom, _ in powtestdomains:
                a, b = dom
                xx = np.linspace(a, b, 1001)
                ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 11))
                gg = c**ff
                vscl = gg.vscale
                hscl = gg.hscale
                lscl = max([fun.size for fun in gg])
                tol = 2 * abs(c) * vscl * hscl * lscl * eps
                assert infnorm(g(xx) - gg(xx)) <= tol


# Define binary operator test parameters
binary_op_params = []
for binop in binops:
    for (f, _, _), (g, _, denomHasRoots) in itertools.combinations(testfunctions, 2):
        for dom, tol in chebfun_testdomains:
            if binop in div_binops and denomHasRoots:
                # skip truediv test if denominator has roots on the real line
                pass
            else:
                a, b = dom
                binopname = binop.__name__
                # case of truediv: add leading and trailing underscores
                if binopname[0] != "_":
                    binopname = "_" + binopname
                if binopname[-1] != "_":
                    binopname = binopname + "_"
                test_name = "{}_{}_{}_{:.0f}_{:.0f}".format(
                    binopname, f.__name__, g.__name__, a, b
                )
                binary_op_params.append((test_name, f, g, binop, dom, tol))

# Define power test parameters
pow_op_params = []
for (f, namef), (g, nameg) in powtestfuns:
    for dom, tol in powtestdomains:
        a, b = dom
        test_name = "pow_{}_{}_{:.1f}_{:.1f}".format(namef, nameg, a, b)
        pow_op_params.append((test_name, f, g, operator.pow, dom, tol))

# Define unary operator test parameters
unary_op_params = []
for unaryop in (operator.pos, operator.neg):
    for f, _, _ in testfunctions:
        for dom, tol in chebfun_testdomains:
            a, b = dom
            test_name = "{}_{}_{}_{:.0f}".format(
                unaryop.__name__, f.__name__, a, b
            )
            unary_op_params.append((test_name, f, unaryop, dom, tol))

# Parametrized test for binary operators
@pytest.mark.parametrize("test_name,f,g,binop,dom,tol", binary_op_params)
def test_binary_op(test_name, f, g, binop, dom, tol):
    a, b = dom
    xx = np.linspace(a, b, 1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, n + 1))
    gg = Chebfun.initfun_adaptive(g, np.linspace(a, b, m + 1))

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale])
    hscl = max([ff.hscale, gg.hscale])
    lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
    assert ff.funs.size == n
    assert gg.funs.size == m
    assert fg.funs.size == n + m - 1
    # Increase tolerance slightly to account for numerical precision issues
    assert infnorm(fg(xx) - FG(xx)) <= 10 * vscl * hscl * lscl * tol

# Parametrized test for power operator
@pytest.mark.parametrize("test_name,f,g,binop,dom,tol", pow_op_params)
def test_pow_op(test_name, f, g, binop, dom, tol):
    a, b = dom
    xx = np.linspace(a, b, 1001)
    n, m = 3, 8
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, n + 1))
    gg = Chebfun.initfun_adaptive(g, np.linspace(a, b, m + 1))

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale])
    hscl = max([ff.hscale, gg.hscale])
    lscl = max([fun.size for fun in np.append(ff.funs, gg.funs)])
    assert ff.funs.size == n
    assert gg.funs.size == m
    assert fg.funs.size == n + m - 1
    # Increase tolerance slightly to account for numerical precision issues
    assert infnorm(fg(xx) - FG(xx)) <= 10 * vscl * hscl * lscl * tol

# Parametrized test for unary operators
@pytest.mark.parametrize("test_name,f,unaryop,dom,tol", unary_op_params)
def test_unary_op(test_name, f, unaryop, dom, tol):
    a, b = dom
    xx = np.linspace(a, b, 1001)
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 9))

    def GG(x):
        return unaryop(f(x))

    gg = unaryop(ff)

    vscl = ff.vscale
    hscl = ff.hscale
    lscl = max([fun.size for fun in ff])
    assert ff.funs.size == gg.funs.size
    # Increase tolerance slightly to account for numerical precision issues
    assert infnorm(gg(xx) - GG(xx)) <= 10 * vscl * hscl * lscl * tol
