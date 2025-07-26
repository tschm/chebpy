"""Unit-tests for chebpy/core/bndfun.py"""

import itertools
import operator
import pytest

import numpy as np

from chebpy.core.bndfun import Bndfun
from chebpy.core.chebtech import Chebtech2
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import Interval
from chebpy.core.algorithms import standard_chop
from chebpy.core.plotting import import_plt

from .utilities import testfunctions, infnorm, joukowsky

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

# NOTE: since (Fun/ClassicFun/)Bndfun is not a user-facing class (although it
# is not abstract) we will test the interface in the way Chebfun will interact
# with it, which means working explcitly with Interval objects. Furthermore,
# since we have already tested the adaptive constructor in the Chebtech-level
# tests, we just use the adaptive constructor in these tests.


"""Unit-tests for miscellaneous Bndfun class usage"""

@pytest.fixture
def class_usage_data():
    subinterval = Interval(-2, 3)
    f = lambda x: sin(30 * x)
    ff = Bndfun.initfun_adaptive(f, subinterval)
    xx = subinterval(np.linspace(-1, 1, 100))
    emptyfun = Bndfun(Chebtech2.initempty(), subinterval)
    constfun = Bndfun(Chebtech2.initconst(1.0), subinterval)
    return {"f": f, "ff": ff, "xx": xx, "emptyfun": emptyfun, "constfun": constfun}

# tests for emptiness of Bndfun objects
def test_isempty_True(class_usage_data):
    emptyfun = class_usage_data["emptyfun"]
    assert emptyfun.isempty
    assert not (not emptyfun.isempty)

def test_isempty_False(class_usage_data):
    constfun = class_usage_data["constfun"]
    assert not constfun.isempty

# tests for constantness of Bndfun objects
def test_isconst_True(class_usage_data):
    constfun = class_usage_data["constfun"]
    assert constfun.isconst
    assert not (not constfun.isconst)

def test_isconst_False(class_usage_data):
    emptyfun = class_usage_data["emptyfun"]
    assert not emptyfun.isconst

# check the size() method is working properly
def test_size():
    cfs = np.random.rand(10)
    subinterval = Interval()
    b0 = Bndfun(Chebtech2(np.array([])), subinterval)
    b1 = Bndfun(Chebtech2(np.array([1.0])), subinterval)
    b2 = Bndfun(Chebtech2(cfs), subinterval)
    assert b0.size == 0
    assert b1.size == 1
    assert b2.size == cfs.size

def test_support(class_usage_data):
    ff = class_usage_data["ff"]
    a, b = ff.support
    assert a == -2
    assert b == 3

def test_endvalues(class_usage_data):
    f = class_usage_data["f"]
    ff = class_usage_data["ff"]
    a, b = ff.support
    fa, fb = ff.endvalues
    assert abs(fa - f(a)) <= 2e1 * eps
    assert abs(fb - f(b)) <= 2e1 * eps

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
    assert infnorm(b - c) <= 2e2 * eps

def test_call_raises(class_usage_data):
    ff = class_usage_data["ff"]
    xx = class_usage_data["xx"]
    with pytest.raises(ValueError):
        ff(xx, "notamethod")
    with pytest.raises(ValueError):
        ff(xx, how="notamethod")

def test_vscale_empty(class_usage_data):
    emptyfun = class_usage_data["emptyfun"]
    assert emptyfun.vscale == 0.0

def test_copy(class_usage_data):
    ff = class_usage_data["ff"]
    gg = ff.copy()
    assert ff == ff
    assert gg == gg
    assert ff != gg
    assert infnorm(ff.coeffs - gg.coeffs) == 0

# check that the restricted fun matches self on the subinterval
def test_restrict(class_usage_data):
    ff = class_usage_data["ff"]
    i1 = Interval(-1, 1)
    gg = ff.restrict(i1)
    yy = np.linspace(-1, 1, 1000)
    assert infnorm(ff(yy) - gg(yy)) <= 1e2 * eps

# check that the restricted fun matches self on the subinterval
def test_simplify(class_usage_data):
    f = class_usage_data["f"]
    interval = Interval(-2, 1)
    ff = Bndfun.initfun_fixedlen(f, interval, 1000)
    gg = ff.simplify()
    assert gg.size == standard_chop(ff.onefun.coeffs)
    assert infnorm(ff.coeffs[: gg.size] - gg.coeffs) == 0
    assert ff.interval == gg.interval

def test_translate(class_usage_data):
    ff = class_usage_data["ff"]
    c = -1
    shifted_interval = ff.interval + c
    gg = ff.translate(c)
    hh = Bndfun.initfun_fixedlen(lambda x: ff(x - c), shifted_interval, gg.size)
    yk = shifted_interval(np.linspace(-1, 1, 100))
    assert gg.interval == hh.interval
    assert infnorm(gg.coeffs - hh.coeffs) <= 2e1 * eps
    assert infnorm(gg(yk) - hh(yk)) <= 1e2 * eps


# --------------------------------------
#          vscale estimates
# --------------------------------------
vscales = [
    # (function, number of points, vscale)
    (lambda x: sin(4 * pi * x), [-2, 2], 1),
    (lambda x: cos(x), [-10, 1], 1),
    (lambda x: cos(4 * pi * x), [-100, 100], 1),
    (lambda x: exp(cos(4 * pi * x)), [-1, 1], exp(1)),
    (lambda x: cos(3244 * x), [-2, 0], 1),
    (lambda x: exp(x), [-1, 2], exp(2)),
    (lambda x: 1e10 * exp(x), [-1, 1], 1e10 * exp(1)),
    (lambda x: 0 * x + 1.0, [-1e5, 1e4], 1),
]


def definiteIntegralTester(fun, interval, vscale):
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)

    def tester():
        absdiff = abs(ff.vscale - vscale)
        assert absdiff <= 0.1 * vscale

    return tester


for k, args in enumerate(vscales):
    _testfun_ = definiteIntegralTester(*args)
    _testfun_.__name__ = "test_vscale_{:02}".format(k)
    globals()[_testfun_.__name__] = _testfun_


plt = import_plt()


"""Unit-tests for Bndfun plotting methods"""

@pytest.fixture
def plotting_data():
    def f(x):
        return sin(1 * x) + 5e-1 * cos(10 * x) + 5e-3 * sin(100 * x)

    def u(x):
        return np.exp(2 * np.pi * 1j * x)

    subinterval = Interval(-6, 10)
    f0 = Bndfun.initfun_fixedlen(f, subinterval, 1000)
    f1 = Bndfun.initfun_adaptive(f, subinterval)
    f2 = Bndfun.initfun_adaptive(u, Interval(-1, 1))
    return {"f0": f0, "f1": f1, "f2": f2}

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot(plotting_data):
    fig, ax = plt.subplots()
    plotting_data["f0"].plot(ax=ax, color="g", marker="o", markersize=2, linestyle="")

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plot_complex(plotting_data):
    fig, ax = plt.subplots()
    # plot Bernstein ellipses
    for rho in np.arange(1.1, 2, 0.1):
        (np.exp(1j * 0.25 * np.pi) * joukowsky(rho * plotting_data["f2"])).plot(ax=ax)

@pytest.mark.skipif(plt is None, reason="matplotlib not installed")
def test_plotcoeffs(plotting_data):
    fig, ax = plt.subplots()
    plotting_data["f0"].plotcoeffs(ax=ax)
    plotting_data["f1"].plotcoeffs(ax=ax, color="r")


"""Unit-tests for Bndfun calculus operations"""

@pytest.fixture
def calculus_data():
    emptyfun = Bndfun(Chebtech2.initempty(), Interval())
    yy = np.linspace(-1, 1, 2000)
    return {"emptyfun": emptyfun, "yy": yy}

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


# --------------------------------------
#           definite integrals
# --------------------------------------
def_integrals = [
    # (function, interval, integral, tolerance)
    (lambda x: sin(x), [-2, 2], 0.0, 2 * eps),
    (lambda x: sin(4 * pi * x), [-0.1, 0.7], 0.088970317927147, 1e1 * eps),
    (lambda x: cos(x), [-100, 203], 0.426944059057085, 1e3 * eps),
    (lambda x: cos(4 * pi * x), [-1e-1, -1e-3], 0.074682699182803, 2 * eps),
    (lambda x: exp(cos(4 * pi * x)), [-3, 1], 5.064263511008033, 4 * eps),
    (lambda x: cos(3244 * x), [0, 0.4], -3.758628487169980e-05, 5e2 * eps),
    (lambda x: exp(x), [-2, -1], exp(-1) - exp(-2), 2 * eps),
    (lambda x: 1e10 * exp(x), [-1, 2], 1e10 * (exp(2) - exp(-1)), 2e10 * eps),
    (lambda x: 0 * x + 1.0, [-100, 300], 400, eps),
]


def definiteIntegralTester(fun, interval, integral, tol):
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)

    def tester():
        absdiff = abs(ff.sum() - integral)
        assert absdiff <= tol

    return tester


for k, (fun, n, integral, tol) in enumerate(def_integrals):
    _testfun_ = definiteIntegralTester(fun, n, integral, tol)
    _testfun_.__name__ = "test_sum_{:02}".format(k)
    globals()[_testfun_.__name__] = _testfun_

# --------------------------------------
#          indefinite integrals
# --------------------------------------
indef_integrals = [
    # (function, indefinite integral, interval, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: x, [-2, 3], eps),
    (lambda x: x, lambda x: 1 / 2 * x**2, [-5, 0], 4 * eps),
    (lambda x: x**2, lambda x: 1 / 3 * x**3, [1, 10], 2e2 * eps),
    (lambda x: x**3, lambda x: 1 / 4 * x**4, [-1e-2, 4e-1], 2 * eps),
    (lambda x: x**4, lambda x: 1 / 5 * x**5, [-3, -2], 3e2 * eps),
    (lambda x: x**5, lambda x: 1 / 6 * x**6, [-1e-10, 1], 4 * eps),
    (lambda x: sin(x), lambda x: -cos(x), [-10, 22], 3e1 * eps),
    (lambda x: cos(3 * x), lambda x: 1.0 / 3 * sin(3 * x), [-3, 4], 2 * eps),
    (lambda x: exp(x), lambda x: exp(x), [-60, 1], 1e1 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * (3 * eps)),
]


def indefiniteIntegralTester(fun, ifn, interval, tol):
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    gg = Bndfun.initfun_fixedlen(ifn, subinterval, ff.size + 1)
    coeffs = gg.coeffs
    coeffs[0] = coeffs[0] - ifn(np.array([interval[0]]))

    def tester():
        absdiff = infnorm(ff.cumsum().coeffs - coeffs)
        assert absdiff <= tol

    return tester


for k, (fun, dfn, n, tol) in enumerate(indef_integrals):
    _testfun_ = indefiniteIntegralTester(fun, dfn, n, tol)
    _testfun_.__name__ = "test_cumsum_{:02}".format(k)
    globals()[_testfun_.__name__] = _testfun_

# --------------------------------------
#            derivatives
# --------------------------------------
derivatives = [
    #     (function, derivative, number of points, tolerance)
    (lambda x: 0 * x + 1.0, lambda x: 0 * x + 0, [-2, 3], eps),
    (lambda x: x, lambda x: 0 * x + 1, [-5, 0], 2e1 * eps),
    (lambda x: x**2, lambda x: 2 * x, [1, 10], 2e2 * eps),
    (lambda x: x**3, lambda x: 3 * x**2, [-1e-2, 4e-1], 3 * eps),
    (lambda x: x**4, lambda x: 4 * x**3, [-3, -2], 1e3 * eps),
    (lambda x: x**5, lambda x: 5 * x**4, [-1e-10, 1], 4e1 * eps),
    (lambda x: sin(x), lambda x: cos(x), [-10, 22], 5e2 * eps),
    (lambda x: cos(3 * x), lambda x: -3 * sin(3 * x), [-3, 4], 5e2 * eps),
    (lambda x: exp(x), lambda x: exp(x), [-60, 1], 2e2 * eps),
    (lambda x: 1e10 * exp(x), lambda x: 1e10 * exp(x), [-1, 1], 1e10 * 2e2 * eps),
]


def derivativeTester(fun, ifn, interval, tol):
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(fun, subinterval)
    gg = Bndfun.initfun_fixedlen(ifn, subinterval, max(ff.size - 1, 1))

    def tester():
        absdiff = infnorm(ff.diff().coeffs - gg.coeffs)
        assert absdiff <= tol

    return tester


for k, (fun, der, n, tol) in enumerate(derivatives):
    _testfun_ = derivativeTester(fun, der, n, tol)
    _testfun_.__name__ = "test_diff_{:02}".format(k)
    globals()[_testfun_.__name__] = _testfun_


@pytest.fixture
def complex_data():
    z = Bndfun.initfun_adaptive(lambda x: np.exp(np.pi * 1j * x), Interval(-1, 1))
    return {"z": z}

def test_init_empty():
    Bndfun.initempty()

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


"""Unit-tests for construction of Bndfun objects"""

def test_onefun_construction():
    coeffs = np.random.rand(10)
    subinterval = Interval()
    onefun = Chebtech2(coeffs)
    f = Bndfun(onefun, subinterval)
    assert isinstance(f, Bndfun)
    assert infnorm(f.coeffs - coeffs) < eps

def test_const_construction():
    subinterval = Interval()
    ff = Bndfun.initconst(1.0, subinterval)
    assert ff.size == 1
    assert ff.isconst
    assert not ff.isempty
    with pytest.raises(ValueError):
        Bndfun.initconst([1.0], subinterval)

def test_empty_construction():
    ff = Bndfun.initempty()
    assert ff.size == 0
    assert not ff.isconst
    assert ff.isempty
    with pytest.raises(TypeError):
        Bndfun.initempty([1.0])

def test_identity_construction():
    for a, b in [(-1, 1), (-10, -2), (-2.3, 1.24), (20, 2000)]:
        itvl = Interval(a, b)
        ff = Bndfun.initidentity(itvl)
        assert ff.size == 2
        xx = np.linspace(a, b, 1001)
        tol = eps * abs(itvl).max()
        assert infnorm(ff(xx) - xx) <= tol


def adaptiveTester(fun, subinterval, funlen):
    ff = Bndfun.initfun_adaptive(fun, subinterval)

    def tester():
        assert ff.size == funlen

    return tester


def fixedlenTester(fun, subinterval, n):
    ff = Bndfun.initfun_fixedlen(fun, subinterval, n)

    def tester():
        assert ff.size == n

    return tester


funs = []
fun_details = [
    # (function, name for the test printouts,
    #  Matlab chebfun adaptive degree on [-2,3])
    (lambda x: x**3 + x**2 + x + 1, "poly3(x)", [-2, 3], 4),
    (lambda x: exp(x), "exp(x)", [-2, 3], 19),
    (lambda x: sin(x), "sin(x)", [-2, 3], 20),
    (lambda x: cos(20 * x), "cos(20x)", [-2, 3], 90),
    (lambda x: 0.0 * x + 1.0, "constfun", [-2, 3], 1),
    (lambda x: 0.0 * x, "zerofun", [-2, 3], 1),
]

for k, (fun, name, interval, funlen) in enumerate(fun_details):
    fun.__name__ = name
    subinterval = Interval(*interval)

    # add the adaptive tests
    _testfun_ = adaptiveTester(fun, subinterval, funlen)
    _testfun_.__name__ = "test_adaptive_{}".format(fun.__name__)
    globals()[_testfun_.__name__] = _testfun_

    # add the fixedlen tests
    for n in np.array([100]):
        _testfun_ = fixedlenTester(fun, subinterval, n)
        _testfun_.__name__ = "test_fixedlen_{}_{:003}pts".format(fun.__name__, n)
        globals()[_testfun_.__name__] = _testfun_


"""Unit-tests for Bndfun algebraic operations"""

@pytest.fixture
def algebra_data():
    yy = np.linspace(-1, 1, 1000)
    emptyfun = Bndfun.initempty()
    return {"yy": yy, "emptyfun": emptyfun}

# check (empty Bndfun) + (Bndfun) = (empty Bndfun)
#   and (Bndfun) + (empty Bndfun) = (empty Bndfun)
def test__add__radd__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun + chebtech).isempty
        assert (chebtech + emptyfun).isempty

# check the output of (constant + Bndfun)
#                 and (Bndfun + constant)
def test__add__radd__constant(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const + fun(x)

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            f1 = const + bndfun
            f2 = bndfun + const
            tol = 4e1 * eps * abs(const)
            assert infnorm(f(xx) - f1(xx)) <= tol
            assert infnorm(f(xx) - f2(xx)) <= tol

# check (empty Bndfun) - (Bndfun) = (empty Bndfun)
#   and (Bndfun) - (empty Bndfun) = (empty Bndfun)
def test__sub__rsub__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun - chebtech).isempty
        assert (chebtech - emptyfun).isempty

# check the output of constant - Bndfun
#                 and Bndfun - constant
def test__sub__rsub__constant(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const - fun(x)

            def g(x):
                return fun(x) - const

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            ff = const - bndfun
            gg = bndfun - const
            tol = 5e1 * eps * abs(const)
            assert infnorm(f(xx) - ff(xx)) <= tol
            assert infnorm(g(xx) - gg(xx)) <= tol

# check (empty Bndfun) * (Bndfun) = (empty Bndfun)
#   and (Bndfun) * (empty Bndfun) = (empty Bndfun)
def test__mul__rmul__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        chebtech = Bndfun.initfun_adaptive(fun, subinterval)
        assert (emptyfun * chebtech).isempty
        assert (chebtech * emptyfun).isempty

# check the output of constant * Bndfun
#                 and Bndfun * constant
def test__mul__rmul__constant(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, _ in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const * fun(x)

            def g(x):
                return fun(x) * const

            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            ff = const * bndfun
            gg = bndfun * const
            tol = 4e1 * eps * abs(const)
            assert infnorm(f(xx) - ff(xx)) <= tol
            assert infnorm(g(xx) - gg(xx)) <= tol

# check (empty Bndfun) / (Bndfun) = (empty Bndfun)
#   and (Bndfun) / (empty Bndfun) = (empty Bndfun)
def test_truediv_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    subinterval = Interval(-2, 3)
    for fun, _, _ in testfunctions:
        bndfun = Bndfun.initfun_adaptive(fun, subinterval)
        assert operator.truediv(emptyfun, bndfun).isempty
        assert operator.truediv(emptyfun, bndfun).isempty
        # __truediv__
        assert (emptyfun / bndfun).isempty
        assert (bndfun / emptyfun).isempty

# check the output of constant / Bndfun
#                 and Bndfun / constant
def test_truediv_constant(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for fun, _, hasRoots in testfunctions:
        for const in (-1, 1, 10, -1e5):

            def f(x):
                return const / fun(x)

            def g(x):
                return fun(x) / const

            hscl = abs(subinterval).max()
            tol = hscl * eps * abs(const)
            bndfun = Bndfun.initfun_adaptive(fun, subinterval)
            gg = bndfun / const
            assert infnorm(g(xx) - gg(xx)) <= 3 * gg.size * tol
            # don't do the following test for functions with roots
            if not hasRoots:
                ff = const / bndfun
                assert infnorm(f(xx) - ff(xx)) <= 2 * ff.size * tol

# check    +(empty Bndfun) = (empty Bndfun)
def test__pos__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    assert (+emptyfun).isempty

# check -(empty Bndfun) = (empty Bndfun)
def test__neg__empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    assert (-emptyfun).isempty

# check (empty Bndfun) ** c = (empty Bndfun)
def test_pow_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for c in range(10):
        assert (emptyfun**c).isempty

# check c ** (empty Bndfun) = (empty Bndfun)
def test_rpow_empty(algebra_data):
    emptyfun = algebra_data["emptyfun"]
    for c in range(10):
        assert (c**emptyfun).isempty

# check the output of Bndfun ** constant
def test_pow_const(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for func in (np.sin, np.exp, np.cos):
        for c in (1, 2):

            def f(x):
                return func(x) ** c

            ff = Bndfun.initfun_adaptive(func, subinterval) ** c
            tol = 2e1 * eps * abs(c)
            assert infnorm(f(xx) - ff(xx)) <= tol

# check the output of constant ** Bndfun
def test_rpow_const(algebra_data):
    yy = algebra_data["yy"]
    subinterval = Interval(-0.5, 0.9)
    xx = subinterval(yy)
    for func in (np.sin, np.exp, np.cos):
        for c in (1, 2):

            def f(x):
                return c ** func(x)

            ff = c ** Bndfun.initfun_adaptive(func, subinterval)
            tol = 1e1 * eps * abs(c)
            assert infnorm(f(xx) - ff(xx)) <= tol


binops = (operator.add, operator.mul, operator.sub, operator.truediv)


# add tests for the binary operators
def binaryOpTester(f, g, subinterval, binop):
    ff = Bndfun.initfun_adaptive(f, subinterval)
    gg = Bndfun.initfun_adaptive(g, subinterval)

    def FG(x):
        return binop(f(x), g(x))

    fg = binop(ff, gg)

    def tester(algebra_data):
        vscl = max([ff.vscale, gg.vscale])
        lscl = max([ff.size, gg.size])
        yy = algebra_data["yy"]
        xx = subinterval(yy)
        assert infnorm(fg(xx) - FG(xx)) <= 6 * vscl * lscl * eps

    return tester


# Note: defining __radd__(a,b) = __add__(b,a) and feeding this into the
# test will not in fact test the __radd__ functionality of the class.
# These tests will need to be added manually.

subintervals = (
    Interval(-0.5, 0.9),
    Interval(-1.2, 1.3),
    Interval(-2.2, -1.9),
    Interval(0.4, 1.3),
)

for binop in binops:
    # add the generic binary operator tests
    for (f, _, _), (g, _, denomRoots) in itertools.combinations(testfunctions, 2):
        for subinterval in subintervals:
            if binop is operator.truediv and denomRoots:
                # skip truediv test if denominator has roots on the real line
                pass
            else:
                _testfun_ = binaryOpTester(f, g, subinterval, binop)
                a, b = subinterval
                _testfun_.__name__ = "test_{}_{}_{}_[{:.1f},{:.1f}]".format(
                    binop.__name__, f.__name__, g.__name__, a, b
                )
                globals()[_testfun_.__name__] = _testfun_

powtestfuns = (
    [(np.exp, "exp"), (np.sin, "sin")],
    [(np.exp, "exp"), (lambda x: 2 - x, "linear")],
    [(lambda x: 2 - x, "linear"), (np.exp, "exp")],
)
# add operator.power tests
for (f, namef), (g, nameg) in powtestfuns:
    for subinterval in subintervals:
        _testfun_ = binaryOpTester(f, g, subinterval, operator.pow)
        a, b = subinterval
        _testfun_.__name__ = "test_{}_{}_{}_[{:.1f},{:.1f}]".format("pow", namef, nameg, a, b)
        globals()[_testfun_.__name__] = _testfun_

unaryops = (operator.pos, operator.neg)


# add tests for the unary operators
def unaryOpTester(unaryop, f, subinterval):
    ff = Bndfun.initfun_adaptive(f, subinterval)

    def gg(x):
        return unaryop(f(x))

    GG = unaryop(ff)

    def tester(algebra_data):
        yy = algebra_data["yy"]
        xx = subinterval(yy)
        assert infnorm(gg(xx) - GG(xx)) <= 4e1 * eps

    return tester


for unaryop in unaryops:
    for f, _, _ in testfunctions:
        subinterval = Interval(-0.5, 0.9)
        _testfun_ = unaryOpTester(unaryop, f, subinterval)
        _testfun_.__name__ = "test_{}_{}".format(unaryop.__name__, f.__name__)
        globals()[_testfun_.__name__] = _testfun_


"""Unit-tests for Bndfun numpy ufunc overloads"""

@pytest.fixture
def ufuncs_data():
    yy = np.linspace(-1, 1, 1000)
    emptyfun = Bndfun.initempty()
    return {"yy": yy, "emptyfun": emptyfun}


ufuncs = (
    np.absolute,
    np.arccos,
    np.arccosh,
    np.arcsin,
    np.arcsinh,
    np.arctan,
    np.arctanh,
    np.cos,
    np.cosh,
    np.exp,
    np.exp2,
    np.expm1,
    np.log,
    np.log2,
    np.log10,
    np.log1p,
    np.sinh,
    np.sin,
    np.tan,
    np.tanh,
    np.sqrt,
)


# empty-case tests
def ufuncEmptyCaseTester(ufunc):
    def tester(ufuncs_data):
        emptyfun = ufuncs_data["emptyfun"]
        assert getattr(emptyfun, ufunc.__name__)().isempty

    return tester


for ufunc in ufuncs:
    _testfun_ = ufuncEmptyCaseTester(ufunc)
    _testfun_.__name__ = "test_emptycase_{}".format(ufunc.__name__)
    globals()[_testfun_.__name__] = _testfun_

# TODO: Add more test cases
# add ufunc tests:
#     (ufunc, [([fun1, interval1], tol1), ([fun2, interval2], tol2), ... ])


def uf1(x):
    """uf1.__name__ = "x" """
    return x


def uf2(x):
    """uf2.__name__ = "sin(x-.5)" """
    return sin(x - 0.5)


def uf3(x):
    """uf3.__name__ = "sin(25*x-1)" """
    return sin(25 * x - 1)


ufunc_test_params = [
    (
        np.absolute,
        [
            ([uf1, (-3, -0.5)], eps),
        ],
    ),
    (
        np.arccos,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arccosh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arcsin,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arcsinh,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.arctan,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.arctanh,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.exp2,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.log,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log2,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log10,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.log1p,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf1, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf1, (-3, 3)], eps),
        ],
    ),
    (
        np.sqrt,
        [
            ([uf1, (2, 3)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf2, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf2, (-3, 3)], eps),
        ],
    ),
    (
        np.cos,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.cosh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.exp,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.expm1,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.sinh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.sin,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
    (
        np.tan,
        [
            ([uf3, (-0.8, 0.8)], eps),
        ],
    ),
    (
        np.tanh,
        [
            ([uf3, (-3, 3)], eps),
        ],
    ),
]


def ufuncTester(ufunc, f, interval, tol):
    ff = Bndfun.initfun_adaptive(f, interval)

    def gg(x):
        return ufunc(f(x))

    GG = getattr(ff, ufunc.__name__)()

    def tester(ufuncs_data):
        yy = ufuncs_data["yy"]
        xx = interval(yy)
        vscl = GG.vscale
        lscl = GG.size
        assert infnorm(gg(xx) - GG(xx)) <= vscl * lscl * tol

    return tester


for (
    ufunc,
    [
        ([f, intvl], tol),
    ],
) in ufunc_test_params:
    interval = Interval(*intvl)
    _testfun_ = ufuncTester(ufunc, f, interval, tol)
    _testfun_.__name__ = "test_{}_{}_[{:.1f},{:.1f}]".format(ufunc.__name__, f.__name__, *intvl)
    globals()[_testfun_.__name__] = _testfun_


def test_roots_empty():
    ff = Bndfun.initempty()
    assert ff.roots().size == 0

def test_roots_const():
    ff = Bndfun.initconst(0.0, Interval(-2, 3))
    gg = Bndfun.initconst(2.0, Interval(-2, 3))
    assert ff.roots().size == 0
    assert gg.roots().size == 0


# add tests for roots
def rootsTester(f, interval, roots, tol):
    subinterval = Interval(*interval)
    ff = Bndfun.initfun_adaptive(f, subinterval)
    rts = ff.roots()

    def tester():
        assert infnorm(rts - roots) <= tol

    return tester


rootstestfuns = (
    (lambda x: 3 * x + 2.0, [-2, 3], np.array([-2 / 3]), eps),
    (lambda x: x**2 + 0.2 * x - 0.08, [-2, 5], np.array([-0.4, 0.2]), 3e1 * eps),
    (lambda x: sin(x), [-7, 7], pi * np.linspace(-2, 2, 5), 1e1 * eps),
    (lambda x: cos(2 * pi * x), [-20, 10], np.linspace(-19.75, 9.75, 60), 3e1 * eps),
    (lambda x: sin(100 * pi * x), [-0.5, 0.5], np.linspace(-0.5, 0.5, 101), eps),
    (lambda x: sin(5 * pi / 2 * x), [-1, 1], np.array([-0.8, -0.4, 0, 0.4, 0.8]), eps),
)
for k, args in enumerate(rootstestfuns):
    _testfun_ = rootsTester(*args)
    _testfun_.__name__ = "test_roots_{}".format(k)
    globals()[_testfun_.__name__] = _testfun_

# reset the testsfun variable so it doesn't get picked up by nose
_testfun_ = None
