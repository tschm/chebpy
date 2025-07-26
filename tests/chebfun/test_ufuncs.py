"""Unit-tests for chebpy/core/chebfun.py - Ufuncs class"""

import unittest
import numpy as np

from chebpy.core.chebfun import Chebfun
from chebpy.core.settings import DefaultPreferences
from ..utilities import infnorm
from chebpy.core.utilities import Interval

# aliases
eps = DefaultPreferences.eps


class Ufuncs(unittest.TestCase):
    def setUp(self):
        self.emptyfun = Chebfun.initempty()
        self.yy = np.linspace(-1, 1, 2000)

    def test_abs_absolute_alias(self):
        self.assertEqual(Chebfun.abs, Chebfun.absolute)


# Define the ufuncs to test
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
    def tester(self):
        self.assertTrue(getattr(self.emptyfun, ufunc.__name__)().isempty)

    return tester


for ufunc in ufuncs:
    _testfun_ = ufuncEmptyCaseTester(ufunc)
    _testfun_.__name__ = "test_emptycase_{}".format(ufunc.__name__)
    setattr(Ufuncs, _testfun_.__name__, _testfun_)


# Define test functions
def uf1(x):
    """uf1.__name__ = "x" """
    return x


def uf2(x):
    """uf2.__name__ = "sin(x-.5)" """
    return np.sin(x - 0.5)


def uf3(x):
    """uf3.__name__ = "sin(25*x-1)" """
    return np.sin(25 * x - 1)


# Define test parameters for each ufunc
ufunc_test_params = [
    (
        np.absolute,
        [
            ([uf1, (-3, 3)], eps),
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
        np.absolute,
        [
            ([uf2, (-3, 3)], eps),
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
        np.absolute,
        [
            ([uf3, (-3, 3)], eps),
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


# Function to create test methods for ufuncs
def ufuncTester(ufunc, f, interval, tol):
    a, b = interval
    ff = Chebfun.initfun_adaptive(f, np.linspace(a, b, 13))

    def gg(x):
        return ufunc(f(x))

    GG = getattr(ff, ufunc.__name__)()

    def tester(self):
        xx = interval(self.yy)
        vscl = GG.vscale
        lscl = sum([fun.size for fun in GG])
        self.assertLessEqual(infnorm(gg(xx) - GG(xx)), vscl * lscl * tol)

    return tester


# Dynamically add test methods for ufuncs
for (
    ufunc,
    [
        ([f, intvl], tol),
    ],
) in ufunc_test_params:
    interval = Interval(*intvl)
    _testfun_ = ufuncTester(ufunc, f, interval, tol)
    _testfun_.__name__ = "test_{}({})_[{:.1f},..,{:.1f}]".format(
        ufunc.__name__, f.__name__, *intvl
    )
    setattr(Ufuncs, _testfun_.__name__, _testfun_)
