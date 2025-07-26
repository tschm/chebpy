"""Unit-tests for pyfun/utilities.py"""

import pytest
import numpy as np

from chebpy.core.settings import DefaultPreferences
from chebpy.core.chebtech import Chebtech2
from chebpy.core.algorithms import bary, clenshaw, coeffmult

from .utilities import testfunctions, scaled_tol, infNormLessThanTol, infnorm

# aliases
pi = np.pi
sin = np.sin
cos = np.cos
exp = np.exp
eps = DefaultPreferences.eps

np.random.seed(0)

# turn off 'divide' and 'invalid' Runtimewarnings: these are invoked in the
# barycentric formula and the warned-of behaviour is actually required
np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture
def evaluation_data():
    npts = 15
    return {
        "xk": Chebtech2._chebpts(npts),
        "vk": Chebtech2._barywts(npts),
        "fk": np.random.rand(npts),
        "ak": np.random.rand(11),
        "xx": -1 + 2 * np.random.rand(9),
        "pts": -1 + 2 * np.random.rand(1001)
    }

# Tests for the Barycentric formula and Clenshaw algorithm
class TestEvaluation:
    # check an empty array is returned whenever either or both of the first
    # two arguments are themselves empty arrays
    def test_bary__empty(self, evaluation_data):
        pts = evaluation_data["pts"]
        null = (None, None)
        assert bary(np.array([]), np.array([]), *null).size == 0
        assert bary(np.array([0.1]), np.array([]), *null).size == 0
        assert bary(np.array([]), np.array([0.1]), *null).size == 0
        assert bary(pts, np.array([]), *null).size == 0
        assert bary(np.array([]), pts, *null).size == 0
        assert bary(np.array([0.1]), np.array([0.1]), *null).size != 0

    def test_clenshaw__empty(self, evaluation_data):
        pts = evaluation_data["pts"]
        assert clenshaw(np.array([]), np.array([])).size == 0
        assert clenshaw(np.array([]), np.array([1.0])).size == 0
        assert clenshaw(np.array([1.0]), np.array([])).size == 0
        assert clenshaw(pts, np.array([])).size == 0
        assert clenshaw(np.array([]), pts).size == 0
        assert clenshaw(np.array([0.1]), np.array([0.1])).size != 0

    # check that scalars get evaluated to scalars (not arrays)
    def test_clenshaw__scalar_input(self, evaluation_data):
        xx = evaluation_data["xx"]
        ak = evaluation_data["ak"]
        for x in xx:
            assert np.isscalar(clenshaw(x, ak))
        assert not np.isscalar(clenshaw(xx, ak))

    def test_bary__scalar_input(self, evaluation_data):
        xx = evaluation_data["xx"]
        fk = evaluation_data["fk"]
        xk = evaluation_data["xk"]
        vk = evaluation_data["vk"]
        for x in xx:
            assert np.isscalar(bary(x, fk, xk, vk))
        assert not np.isscalar(bary(xx, fk, xk, vk))

    # Check that we always get float output for constant Chebtechs, even
    # when passing in an integer input.
    # TODO: Move these tests elsewhere?
    def test_bary__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.0)
        assert isinstance(ff(0, "bary"), float)
        assert isinstance(gg(0, "bary"), float)

    def test_clenshaw__float_output(self):
        ff = Chebtech2.initconst(1)
        gg = Chebtech2.initconst(1.0)
        assert isinstance(ff(0, "clenshaw"), float)
        assert isinstance(gg(0, "clenshaw"), float)

    # Check that we get consistent output from bary and clenshaw
    # TODO: Move these tests elsewhere?
    def test_bary_clenshaw_consistency(self):
        coeffs = np.random.rand(3)
        evalpts = (0.5, np.array([]), np.array([0.5]), np.array([0.5, 0.6]))
        for n in range(len(coeffs)):
            ff = Chebtech2(coeffs[:n])
            for xx in evalpts:
                fb = ff(xx, "bary")
                fc = ff(xx, "clenshaw")
                assert type(fb) == type(fc)


evalpts = [np.linspace(-1, 1, int(n)) for n in np.array([1e2, 1e3, 1e4, 1e5])]
ptsarry = [Chebtech2._chebpts(n) for n in np.array([100, 200])]
methods = [bary, clenshaw]

# Generate test parameters
test_params = []
for method in methods:
    for fun, _, _ in testfunctions:
        for j, chebpts in enumerate(ptsarry):
            for k, xx in enumerate(evalpts):
                test_params.append((method, fun, xx, chebpts, j, k))

# Define the test function with parametrization
@pytest.mark.parametrize(
    "method,fun,evalpts,chebpts,j,k",
    test_params,
    ids=lambda param: f"{param.__name__}" if hasattr(param, "__name__") else str(param)
)
def test_eval_methods(method, fun, evalpts, chebpts, j, k):
    x = evalpts
    xk = chebpts
    fvals = fun(xk)

    if method is bary:
        vk = Chebtech2._barywts(fvals.size)
        a = bary(x, fvals, xk, vk)
        tol_multiplier = 1e0

    elif method is clenshaw:
        ak = Chebtech2._vals2coeffs(fvals)
        a = clenshaw(x, ak)
        tol_multiplier = 2e1

    b = fun(evalpts)
    n = evalpts.size
    tol = tol_multiplier * scaled_tol(n)

    assert infnorm(a - b) <= tol


@pytest.fixture
def coeffmult_data():
    return {
        "f": lambda x: exp(x),
        "g": lambda x: cos(x),
        "fn": 15,
        "gn": 15
    }

class TestCoeffMult:
    def test_coeffmult(self, coeffmult_data):
        f = coeffmult_data["f"]
        g = coeffmult_data["g"]
        fn = coeffmult_data["fn"]
        gn = coeffmult_data["gn"]

        def h(x):
            return f(x) * g(x)

        hn = fn + gn - 1
        fc = Chebtech2.initfun(f, fn).prolong(hn).coeffs
        gc = Chebtech2.initfun(g, gn).prolong(hn).coeffs
        hc = coeffmult(fc, gc)
        HC = Chebtech2.initfun(h, hn).coeffs
        assert infnorm(hc - HC) <= 2e1 * eps


# reset the testfun variable to avoid any potential issues
testfun = None
