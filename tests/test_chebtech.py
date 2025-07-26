"""Unit-tests for chebpy/core/chebtech.py"""

import itertools
import operator
import pytest

import numpy as np

from chebpy.core.settings import DefaultPreferences
from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import standard_chop
from chebpy.core.plotting import import_plt

from .utilities import testfunctions, infnorm, scaled_tol, infNormLessThanTol, joukowsky

np.random.seed(0)

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps
_vals2coeffs = Chebtech2._vals2coeffs
_coeffs2vals = Chebtech2._coeffs2vals


# ------------------------
"""Unit-tests for Chebtech2"""

def test_chebpts_0():
    assert Chebtech2._chebpts(0).size == 0

def test_vals2coeffs_empty():
    assert _vals2coeffs(np.array([])).size == 0

def test_coeffs2vals_empty():
    assert _coeffs2vals(np.array([])).size == 0

# check we are returned the array for an array of size 1
def test_vals2coeffs_size1():
    for k in np.arange(10):
        fk = np.array([k])
        assert infnorm(_vals2coeffs(fk) - fk) <= eps

# check we are returned the array for an array of size 1
def test_coeffs2vals_size1():
    for k in np.arange(10):
        ak = np.array([k])
        assert infnorm(_coeffs2vals(ak) - ak) <= eps

# TODO: further checks for chepbts


# ------------------------------------------------------------------------
# Tests to verify the mutually inverse nature of vals2coeffs and coeffs2vals
# ------------------------------------------------------------------------
# Generate test parameters for vals2coeffs2vals and coeffs2vals2coeffs tests
test_sizes = 2 ** np.arange(2, 18, 2) + 1

@pytest.mark.parametrize("n", test_sizes)
def test_vals2coeffs2vals(n):
    values = np.random.rand(n)
    coeffs = _vals2coeffs(values)
    _values_ = _coeffs2vals(coeffs)
    assert infnorm(values - _values_) <= scaled_tol(n)

@pytest.mark.parametrize("n", test_sizes)
def test_coeffs2vals2coeffs(n):
    coeffs = np.random.rand(n)
    values = _coeffs2vals(coeffs)
    _coeffs_ = _vals2coeffs(values)
    assert infnorm(coeffs - _coeffs_) <= scaled_tol(n)
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
# Test second-kind Chebyshev points
# ------------------------------------------------------------------------
chebpts2_testlist = [
    (Chebtech2._chebpts(1), np.array([0.0]), eps),
    (Chebtech2._chebpts(2), np.array([-1.0, 1.0]), eps),
    (Chebtech2._chebpts(3), np.array([-1.0, 0.0, 1.0]), eps),
    (Chebtech2._chebpts(4), np.array([-1.0, -0.5, 0.5, 1.0]), 2 * eps),
    (
        Chebtech2._chebpts(5),
        np.array([-1.0, -(2.0 ** (-0.5)), 0.0, 2.0 ** (-0.5), 1.0]),
        eps,
    ),
]

@pytest.mark.parametrize("a,b,tol", chebpts2_testlist)
def test_chebpts_values(a, b, tol):
    assert infnorm(a - b) <= tol


# check the output is of the correct length, the endpoint values are -1
# and 1, respectively, and that the sequence is monotonically increasing
chebpts_len_sizes = 2 ** np.arange(2, 18, 2) + 3

@pytest.mark.parametrize("k", chebpts_len_sizes)
def test_chebpts_len(k):
    pts = Chebtech2._chebpts(k)
    assert pts.size == k
    assert pts[0] == -1.0
    assert pts[-1] == 1.0
    assert np.all(np.diff(pts) > 0)
# ------------------------------------------------------------------------


@pytest.fixture
def class_usage_data():
    return {
        "ff": Chebtech2.initfun_fixedlen(lambda x: np.sin(30 * x), 100),
        "xx": -1 + 2 * np.random.rand(100)
    }

"""Unit-tests for miscelaneous Chebtech2 class usage"""

# tests for emptiness of Chebtech2 objects
def test_isempty_True():
    f = Chebtech2(np.array([]))
    assert f.isempty
    assert not (not f.isempty)

def test_isempty_False():
    f = Chebtech2(np.array([1.0]))
    assert not f.isempty

# tests for constantness of Chebtech2 objects
def test_isconst_True():
    f = Chebtech2(np.array([1.0]))
    assert f.isconst
    assert not (not f.isconst)

def test_isconst_False():
    f = Chebtech2(np.array([]))
    assert not f.isconst

# check the size() method is working properly
def test_size():
    cfs = np.random.rand(10)
    assert Chebtech2(np.array([])).size == 0
    assert Chebtech2(np.array([1.0])).size == 1
    assert Chebtech2(cfs).size == cfs.size

# test the different permutations of self(xx, ..)
def test_call(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    ff(xx)

def test_call_bary(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    ff(xx, "bary")
    ff(xx, how="bary")

def test_call_clenshaw(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    ff(xx, "clenshaw")
    ff(xx, how="clenshaw")

def test_call_bary_vs_clenshaw(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    b = ff(xx, "clenshaw")
    c = ff(xx, "bary")
    assert infnorm(b - c) <= 5e1 * eps

def test_call_raises(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    with pytest.raises(ValueError):
        ff(xx, "notamethod")
    with pytest.raises(ValueError):
        ff(xx, how="notamethod")

def test_prolong(class_usage_data):
    ff = class_usage_data["ff"]
    for k in [0, 1, 20, ff.size, 200]:
        assert ff.prolong(k).size == k

def test_vscale_empty():
    gg = Chebtech2(np.array([]))
    assert gg.vscale == 0.0

def test_copy(class_usage_data):
    ff = class_usage_data["ff"]
    gg = ff.copy()
    assert ff == ff
    assert gg == gg
    assert ff != gg
    assert infnorm(ff.coeffs - gg.coeffs) == 0

def test_simplify(class_usage_data):
    ff = class_usage_data["ff"]
    gg = ff.simplify()
    # check that simplify is calling standard_chop underneath
    assert gg.size == standard_chop(ff.coeffs)
    assert infnorm(ff.coeffs[: gg.size] - gg.coeffs) == 0
    # check we are returned a copy of self's coeffcients by changing
    # one entry of gg
    fcfs = ff.coeffs
    gcfs = gg.coeffs
    assert (fcfs[: gg.size] - gcfs).sum() == 0
    gg.coeffs[0] = 1
    assert (fcfs[: gg.size] - gcfs).sum() != 0


# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4 * pi * x), 40, 1),
    (lambda x: cos(x), 15, 1),
    (lambda x: cos(4 * pi * x), 39, 1),
    (lambda x: exp(cos(4 * pi * x)), 181, exp(1)),
    (lambda x: cos(3244 * x), 3389, 1),
    (lambda x: exp(x), 15, exp(1)),
    (lambda x: 1e10 * exp(x), 15, 1e10 * exp(1)),
    (lambda x: 0 * x + 1.0, 1, 1),
]

@pytest.mark.parametrize("fun,n,vscale", vscales)
def test_vscale(fun, n, vscale):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    absdiff = abs(ff.vscale - vscale)
    assert absdiff <= 0.1 * vscale


plt = import_plt()


@pytest.fixture
def plotting_data():
    def f(x):
        return sin(3 * x) + 5e-1 * cos(30 * x)

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    return {
        "f0": Chebtech2.initfun_fixedlen(f, 100),
        "f1": Chebtech2.initfun_adaptive(f),
        "f2": Chebtech2.initfun_adaptive(u)
    }


"""Unit-tests for Chebtech2 plotting methods"""

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot(plotting_data):
    fig, ax = plt.subplots()
    plotting_data["f0"].plot(ax=ax)

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_complex(plotting_data):
    fig, ax = plt.subplots()
    # plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        joukowsky(rho * plotting_data["f2"]).plot(ax=ax)

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plotcoeffs(plotting_data):
    fig, ax = plt.subplots()
    plotting_data["f0"].plotcoeffs(ax=ax)
    plotting_data["f1"].plotcoeffs(ax=ax, color="r")


@pytest.fixture
def calculus_data():
    return {
        "emptyfun": Chebtech2(np.array([]))
    }

"""Unit-tests for Chebtech2 calculus operations"""

# tests for the correct results in the empty cases
def test_sum_empty(calculus_data):
    emptyfun = calculus_data["emptyfun"]
    assert emptyfun.sum() == 0

def test_cumsum_empty(calculus_data):
    emptyfun = calculus_data["emptyfun"]
    assert emptyfun.cumsum().isempty

def test_diff_empty(calculus_data):
    emptyfun = calculus_data["emptyfun"]
    assert emptyfun.diff().isempty


@pytest.fixture
def complex_data():
    return {
        "z": Chebtech2.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x))
    }

def test_init_empty():
    Chebtech2.initempty()

def test_roots(complex_data):
    z = complex_data["z"]
    r0 = z.roots()
    r1 = (z - 1).roots()
    r2 = (z - 1j).roots()
    r3 = (z + 1).roots()
    r4 = (z + 1j).roots()
    assert r0.size == 0
    assert np.allclose(r1, [0])
    assert np.allclose(r2, [0.5])
    assert np.allclose(r3, [-1, 1])
    assert np.allclose(r4, [-0.5])

def test_rho_ellipse_construction(complex_data):
    z = complex_data["z"]
    zz = 1.2 * z
    e = 0.5 * (zz + 1 / zz)
    assert abs(e(1) - e(-1)) < 1e-14
    assert abs(e(0) + e(-1)) < 1e-14
    assert abs(e(0) + e(1)) < 1e-14

def test_calculus(complex_data):
    z = complex_data["z"]
    assert np.allclose([z.sum()], [0])
    assert (z.cumsum().diff() - z).size == 1
    assert (z - z.cumsum().diff()).size == 1

def test_real_imag(complex_data):
    z = complex_data["z"]
    # check definition of real and imaginary
    zreal = z.real()
    zimag = z.imag()
    np.testing.assert_equal(zreal.coeffs, np.real(z.coeffs))
    np.testing.assert_equal(zimag.coeffs, np.imag(z.coeffs))
    # check real part of real chebtech is the same chebtech
    assert zreal.real() == zreal
    # check imaginary part of real chebtech is the zero chebtech
    assert zreal.imag().isconst
    assert zreal.imag().coeffs[0] == 0


# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, number of points, integral, tolerance)
    (lambda x: sin(x), 14, 0.0, eps),
    (lambda x: sin(4 * pi * x), 40, 0.0, 1e1 * eps),
    (lambda x: cos(x), 15, 1.682941969615793, 2 * eps),
    (lambda x: cos(4 * pi * x), 39, 0.0, 2 * eps),
    (lambda x: exp(cos(4 * pi * x)), 182, 2.532131755504016, 4 * eps),
    (lambda x: cos(3244 * x), 3389, 5.879599674161602e-04, 5e2 * eps),
    (lambda x: exp(x), 15, exp(1) - exp(-1), 2 * eps),
    (lambda x: 1e10 * exp(x), 15, 1e10 * (exp(1) - exp(-1)), 4e10 * eps),
    (lambda x: 0 * x + 1.0, 1, 2, eps),
]

@pytest.mark.parametrize("fun,n,integral,tol", def_integrals)
def test_definite_integral(fun, n, integral, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    absdiff = abs(ff.sum() - integral)
    assert absdiff <= tol

# --------------------------------------
#          indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, indefinite integral, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: x, 1, eps),
    (lambda x: x, lambda x: 1 / 2 * x**2, 2, 2 * eps),
    (lambda x: x**2, lambda x: 1 / 3 * x**3, 3, 2 * eps),
    (lambda x: x**3, lambda x: 1 / 4 * x**4, 4, 2 * eps),
    (lambda x: x**4, lambda x: 1 / 5 * x**5, 5, 2 * eps),
    (lambda x: x**5, lambda x: 1 / 6 * x**6, 6, 4 * eps),
    (lambda x: sin(x), lambda x: -cos(x), 16, 2 * eps),
    (lambda x: cos(3 * x), lambda x: 1.0 / 3 * sin(3 * x), 23, 2 * eps),
    (lambda x: exp(x), lambda x: exp(x), 16, 3 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), 16, 1e10 * (3 * eps)),
]

@pytest.mark.parametrize("fun,dfn,n,tol", indef_integrals)
def test_indefinite_integral(fun, dfn, n, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(dfn, n + 1)
    coeffs = gg.coeffs
    coeffs[0] = coeffs[0] - dfn(np.array([-1]))

    absdiff = infnorm(ff.cumsum().coeffs - coeffs)
    assert absdiff <= tol

# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
    # (function, derivative, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: 0 * x + 0, 1, eps),
    (lambda x: x, lambda x: 0 * x + 1, 2, 2 * eps),
    (lambda x: x**2, lambda x: 2 * x, 3, 2 * eps),
    (lambda x: x**3, lambda x: 3 * x**2, 4, 2 * eps),
    (lambda x: x**4, lambda x: 4 * x**3, 5, 3 * eps),
    (lambda x: x**5, lambda x: 5 * x**4, 6, 4 * eps),
    (lambda x: sin(x), lambda x: cos(x), 16, 5e1 * eps),
    (lambda x: cos(3 * x), lambda x: -3 * sin(3 * x), 23, 5e2 * eps),
    (lambda x: exp(x), lambda x: exp(x), 16, 2e2 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), 16, 1e10 * 2e2 * eps),
]

@pytest.mark.parametrize("fun,der,n,tol", derivatives)
def test_derivative(fun, der, n, tol):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    gg = Chebtech2.initfun_fixedlen(der, max(n - 1, 1))

    absdiff = infnorm(ff.diff().coeffs - gg.coeffs)
    assert absdiff <= tol


"""Unit-tests for construction of Chebtech2 objects"""

# TODO: expand to all the constructor variants
def test_initvalues():
    # test n = 0 case separately
    vals = np.random.rand(0)
    fun = Chebtech2.initvalues(vals)
    cfs = Chebtech2._vals2coeffs(vals)
    assert fun.coeffs.size == cfs.size == 0
    # now test the other cases
    for n in range(1, 10):
        vals = np.random.rand(n)
        fun = Chebtech2.initvalues(vals)
        cfs = Chebtech2._vals2coeffs(vals)
        assert infnorm(fun.coeffs - cfs) == 0.0

def test_initidentity():
    x = Chebtech2.initidentity()
    s = -1 + 2 * np.random.rand(10000)
    assert infnorm(s - x(s)) == 0.0

def test_coeff_construction():
    coeffs = np.random.rand(10)
    f = Chebtech2(coeffs)
    assert isinstance(f, Chebtech2)
    assert infnorm(f.coeffs - coeffs) < eps

def test_const_construction():
    ff = Chebtech2.initconst(1.0)
    assert ff.size == 1
    assert ff.isconst
    assert not ff.isempty
    with pytest.raises(ValueError):
        Chebtech2.initconst([1.0])

def test_empty_construction():
    ff = Chebtech2.initempty()
    assert ff.size == 0
    assert not ff.isconst
    assert ff.isempty
    with pytest.raises(TypeError):
        Chebtech2.initempty([1.0])


# Test adaptive function initialization
@pytest.mark.parametrize("fun,funlen,_", testfunctions)
def test_adaptive(fun, funlen, _):
    ff = Chebtech2.initfun_adaptive(fun)
    assert ff.size - funlen <= 2
    assert ff.size - funlen > -1

# Test fixed length function initialization
@pytest.mark.parametrize("fun,funlen,_", testfunctions)
@pytest.mark.parametrize("n", [50, 500])
def test_fixedlen(fun, funlen, _, n):
    ff = Chebtech2.initfun_fixedlen(fun, n)
    assert ff.size == n


@pytest.fixture
def algebra_data():
    return {
        "xx": -1 + 2 * np.random.rand(1000),
        "emptyfun": Chebtech2.initempty()
    }

"""Unit-tests for Chebtech2 algebraic operations"""

# check (empty Chebtech) + (Chebtech) = (empty Chebtech)
#   and (Chebtech) + (empty Chebtech) = (empty Chebtech)
def test__add__radd__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for fun, funlen, _ in testfunctions:
        chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        assert (emptyfun + chebtech).isempty
        assert (chebtech + emptyfun).isempty

# check the output of (constant + Chebtech)
#                 and (Chebtech + constant)
def test__add__radd__constant(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const + fun(x)

            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            f1 = const + techfun
            f2 = techfun + const
            tol = 5e1 * eps * abs(const)
            assert infnorm(f(xx) - f1(xx)) <= tol
            assert infnorm(f(xx) - f2(xx)) <= tol

# check the output of (Chebtech - Chebtech)
def test__add__negself(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen, _ in testfunctions:
        chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        chebzero = chebtech - chebtech
        assert chebzero.isconst
        assert infnorm(chebzero(xx)) == 0

# check (empty Chebtech) - (Chebtech) = (empty Chebtech)
#   and (Chebtech) - (empty Chebtech) = (empty Chebtech)
def test__sub__rsub__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for fun, funlen, _ in testfunctions:
        chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        assert (emptyfun - chebtech).isempty
        assert (chebtech - emptyfun).isempty

# check the output of constant - Chebtech
#                 and Chebtech - constant
def test__sub__rsub__constant(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const - fun(x)

            def g(x):
                return fun(x) - const

            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            ff = const - techfun
            gg = techfun - const
            tol = 5e1 * eps * abs(const)
            assert infnorm(f(xx) - ff(xx)) <= tol
            assert infnorm(g(xx) - gg(xx)) <= tol

# check (empty Chebtech) * (Chebtech) = (empty Chebtech)
#   and (Chebtech) * (empty Chebtech) = (empty Chebtech)
def test__mul__rmul__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for fun, funlen, _ in testfunctions:
        chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        assert (emptyfun * chebtech).isempty
        assert (chebtech * emptyfun).isempty

# check the output of constant * Chebtech
#                 and Chebtech * constant
def test__mul__rmul__constant(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const * fun(x)

            def g(x):
                return fun(x) * const

            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            ff = const * techfun
            gg = techfun * const
            tol = 5e1 * eps * abs(const)
            assert infnorm(f(xx) - ff(xx)) <= tol
            assert infnorm(g(xx) - gg(xx)) <= tol

# check (empty Chebtech) / (Chebtech) = (empty Chebtech)
#   and (Chebtech) / (empty Chebtech) = (empty Chebtech)
def test_truediv_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for fun, funlen, _ in testfunctions:
        chebtech = Chebtech2.initfun_fixedlen(fun, funlen)
        assert operator.truediv(emptyfun, chebtech).isempty
        assert operator.truediv(chebtech, emptyfun).isempty
        # __truediv__
        assert (emptyfun / chebtech).isempty
        assert (chebtech / emptyfun).isempty

# check the output of constant / Chebtech
#                 and Chebtech / constant
# this tests truediv, __rdiv__, __truediv__, __rtruediv__, since
# from __future__ import division is executed at the top of the file
# TODO: find a way to test truediv and  __truediv__ genuinely separately
def test_truediv_constant(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen, hasRoots in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const / fun(x)

            def g(x):
                return fun(x) / const

            tol = eps * abs(const)
            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            gg = techfun / const
            assert infnorm(g(xx) - gg(xx)) <= 2 * gg.size * tol
            # don't do the following test for functions with roots
            if not hasRoots:
                ff = const / techfun
                assert infnorm(f(xx) - ff(xx)) <= 3 * ff.size * tol

# check    +(empty Chebtech) = (empty Chebtech)
def test__pos__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    assert (+emptyfun).isempty

# check -(empty Chebtech) = (empty Chebtech)
def test__neg__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    assert (-emptyfun).isempty

def test_pow_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for c in range(10):
        assert (emptyfun**c).isempty

def test_rpow_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for c in range(10):
        assert (c**emptyfun).isempty

# check the output of Chebtech ** constant
#                 and constant ** Chebtech
def test_pow_const(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen in [(np.sin, 15), (np.exp, 15)]:
        for c in (1, 2):

            def f(x):
                return fun(x) ** c

            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            ff = techfun**c
            tol = 2e1 * eps * abs(c)
            assert infnorm(f(xx) - ff(xx)) <= tol

def test_rpow_const(algebra_data):
    xx = algebra_data["xx"]
    for fun, funlen in [(np.sin, 15), (np.exp, 15)]:
        for c in (1, 2):

            def g(x):
                return c ** fun(x)

            techfun = Chebtech2.initfun_fixedlen(fun, funlen)
            gg = c**techfun
            tol = 2e1 * eps * abs(c)
            assert infnorm(g(xx) - gg(xx)) <= tol


# Generate test parameters for binary operators
binary_op_params = []
binops = (operator.add, operator.mul, operator.sub, operator.truediv)
for binop in binops:
    # add generic binary operator tests
    for (f, nf, _), (g, ng, denomRoots) in itertools.combinations(testfunctions, 2):
        if binop is operator.truediv and denomRoots:
            # skip truediv test if the denominator has roots
            pass
        else:
            binary_op_params.append((f, g, binop, nf, ng, f.__name__, g.__name__, binop.__name__))

# Add power operator tests
powtestfuns = (
    [(np.exp, 15, "exp"), (np.sin, 15, "sin")],
    [(np.exp, 15, "exp"), (lambda x: 2 - x, 2, "linear")],
    [(lambda x: 2 - x, 2, "linear"), (np.exp, 15, "exp")],
)
for (f, nf, namef), (g, ng, nameg) in powtestfuns:
    binary_op_params.append((f, g, operator.pow, nf, ng, namef, nameg, "pow"))

# Test binary operators
@pytest.mark.parametrize("f,g,binop,nf,ng,fname,gname,binopname", binary_op_params)
def test_binary_operators(f, g, binop, nf, ng, fname, gname, binopname, algebra_data):
    xx = algebra_data["xx"]
    ff = Chebtech2.initfun_fixedlen(f, nf)
    gg = Chebtech2.initfun_fixedlen(g, ng)

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    vscl = max([ff.vscale, gg.vscale])
    lscl = max([ff.size, gg.size])
    assert infnorm(fg(xx) - FG(xx)) <= 3 * vscl * lscl * eps

    if binop is operator.mul:
        # check simplify is not being called in __mul__
        assert fg.size == ff.size + gg.size - 1


# Generate test parameters for unary operators
unary_op_params = []
unaryops = (operator.pos, operator.neg)
for unaryop in unaryops:
    for f, nf, _ in testfunctions:
        unary_op_params.append((unaryop, f, nf, unaryop.__name__, f.__name__))

# Test unary operators
@pytest.mark.parametrize("unaryop,f,nf,unaryopname,fname", unary_op_params)
def test_unary_operators(unaryop, f, nf, unaryopname, fname, algebra_data):
    xx = algebra_data["xx"]
    ff = Chebtech2.initfun_fixedlen(f, nf)

    def gg(x):
        return unaryop(f(x))

    GG = unaryop(ff)

    assert infnorm(gg(xx) - GG(xx)) <= 4e1 * eps


def test_roots_empty():
    ff = Chebtech2.initempty()
    assert ff.roots().size == 0

def test_roots_const():
    ff = Chebtech2.initconst(0.0)
    gg = Chebtech2.initconst(2.0)
    assert ff.roots().size == 0
    assert gg.roots().size == 0


# Test roots
rootstestfuns = [
    (lambda x: 3 * x + 2.0, np.array([-2 / 3]), 1 * eps),
    (lambda x: x**2, np.array([0.0, 0.0]), 1 * eps),
    (lambda x: x**2 + 0.2 * x - 0.08, np.array([-0.4, 0.2]), 1 * eps),
    (lambda x: sin(x), np.array([0]), 1 * eps),
    (lambda x: cos(2 * pi * x), np.array([-0.75, -0.25, 0.25, 0.75]), 1 * eps),
    (lambda x: sin(100 * pi * x), np.linspace(-1, 1, 201), 1 * eps),
    (lambda x: sin(5 * pi / 2 * x), np.array([-0.8, -0.4, 0, 0.4, 0.8]), 1 * eps),
]

@pytest.mark.parametrize("f,roots,tol", rootstestfuns)
def test_roots(f, roots, tol):
    ff = Chebtech2.initfun_adaptive(f)
    rts = ff.roots()
    assert infnorm(rts - roots) <= tol
