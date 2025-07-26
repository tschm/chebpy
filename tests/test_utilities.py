"""Unit-tests for pyfun/core/utilities.py"""

import pytest
import numpy as np

from chebpy import chebfun
from chebpy.core.bndfun import Bndfun
from chebpy.core.settings import DefaultPreferences
from chebpy.core.utilities import HTOL, Interval, Domain, compute_breakdata, check_funs
from chebpy.core.exceptions import (
    IntervalGap,
    IntervalOverlap,
    IntervalValues,
    InvalidDomain,
    SupportMismatch,
    NotSubdomain,
)

from .utilities import infnorm


np.random.seed(0)
eps = DefaultPreferences.eps
HTOL = HTOL()


@pytest.fixture
def interval_data():
    return {
        "i1": Interval(-2, 3),
        "i2": Interval(-2, 3),
        "i3": Interval(-1, 1),
        "i4": Interval(-1, 2)
    }

# tests for usage of the Interval class
class TestInterval:
    def test_init(self):
        Interval(-1, 1)
        assert (np.asarray(Interval()) == np.array([-1, 1])).all()

    def test_init_disallow(self):
        with pytest.raises(IntervalValues):
            Interval(2, 0)
        with pytest.raises(IntervalValues):
            Interval(0, 0)

    def test__eq__(self, interval_data):
        i1 = interval_data["i1"]
        i2 = interval_data["i2"]
        i3 = interval_data["i3"]
        assert Interval() == Interval()
        assert i1 == i2
        assert i2 == i1
        assert not (i3 == i1)
        assert not (i2 == i3)

    def test__ne__(self, interval_data):
        i1 = interval_data["i1"]
        i2 = interval_data["i2"]
        i3 = interval_data["i3"]
        assert not (Interval() != Interval())
        assert not (i1 != i2)
        assert not (i2 != i1)
        assert i3 != i1
        assert i2 != i3

    def test__contains__(self, interval_data):
        i1 = interval_data["i1"]
        i2 = interval_data["i2"]
        i3 = interval_data["i3"]
        i4 = interval_data["i4"]
        assert i1 in i2
        assert i3 in i1
        assert i4 in i1
        assert not (i1 in i3)
        assert not (i1 in i4)
        assert not (i1 not in i2)
        assert not (i3 not in i1)
        assert not (i4 not in i1)
        assert i1 not in i3
        assert i1 not in i4

    # Interval objects used to have tolerance-sensitive definitions of __eq__ and
    # __contains__, though these were removed in the commit following
    # 9eaf1c5e0674dab1a676d04a02ceda329beec2ea.
    #    def test__eq__close(self):
    #        tol = .8*HTOL
    #        i4 = Interval(-2,5)
    #        i5 = Interval(-2*(1+tol),5*(1-tol))
    #        i6 = Interval(-2*(1+2*tol),5*(1-2*tol))
    #        assert i4 == i5
    #        assert i4 != i6

    #    def test__contains__close(self):
    #        tol = .8*HTOL
    #        i1 = Interval(-1,2)
    #        i2 = Interval(-1-tol,2+2*tol)
    #        i3 = Interval(-1-2*tol,2+4*tol)
    #        assert i1 in i2
    #        assert i2 in i1
    #        assert not (i3 in i1)

    def test_maps(self):
        yy = -1 + 2 * np.random.rand(1000)
        interval = Interval(-2, 3)
        vals = interval.invmap(interval(yy)) - yy
        assert infnorm(vals) <= eps

    def test_isinterior(self):
        npts = 1000
        x1 = np.linspace(-2, 3, npts)
        x2 = np.linspace(-3, -2, npts)
        x3 = np.linspace(3, 4, npts)
        x4 = np.linspace(5, 6, npts)
        interval = Interval(-2, 3)
        assert interval.isinterior(x1).sum() == npts - 2
        assert interval.isinterior(x2).sum() == 0
        assert interval.isinterior(x3).sum() == 0
        assert interval.isinterior(x4).sum() == 0


# tests for usage of the Domain class
class TestDomain:
    def test__init__(self):
        Domain([-2, 1])
        Domain([-2, 0, 1])
        Domain(np.array([-2, 1]))
        Domain(np.array([-2, 0, 1]))
        Domain(np.linspace(-10, 10, 51))

    def test__init__disallow(self):
        with pytest.raises(InvalidDomain):
            Domain([1])
        with pytest.raises(InvalidDomain):
            Domain([1, -1])
        with pytest.raises(InvalidDomain):
            Domain([-1, 0, 0])
        with pytest.raises(ValueError):
            Domain(["a", "b"])

    def test__iter__(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain([-1, 0, 1, 2])
        res_a = (-2, 1)
        res_b = (-2, 0, 1)
        res_c = (-1, 0, 1, 2)
        assert all([x == y for x, y in zip(dom_a, res_a)])
        assert all([x == y for x, y in zip(dom_b, res_b)])
        assert all([x == y for x, y in zip(dom_c, res_c)])

    def test_intervals(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain([-1, 0, 1, 2])
        res_a = [(-2, 1)]
        res_b = [(-2, 0), (0, 1)]
        res_c = [(-1, 0), (0, 1), (1, 2)]
        assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_a.intervals, res_a)])
        assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_b.intervals, res_b)])
        assert all([itvl == Interval(a, b) for itvl, (a, b) in zip(dom_c.intervals, res_c)])

    def test__contains__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-1, 2])
        d3 = Domain(np.linspace(-10, 10, 1000))
        d4 = Domain([-1, 0, 1, 2])
        assert d2 in d1
        assert d1 in d3
        assert d2 in d3
        assert d2 in d3
        assert d2 in d4
        assert d4 in d2
        assert not (d1 in d2)
        assert not (d3 in d1)
        assert not (d3 in d2)

    def test__contains__close(self):
        tol = 0.8 * HTOL
        d1 = Domain([-1, 2])
        d2 = Domain([-1 - tol, 2 + 2 * tol])
        d3 = Domain([-1 - 2 * tol, 2 + 4 * tol])
        assert d1 in d2
        assert d2 in d1
        assert not (d3 in d1)

    def test__eq__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        assert d1 == d2
        assert d1 != d3

    def test__eq___result_type(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        assert isinstance(d1 == d2, bool)
        assert isinstance(d1 == d3, bool)

    def test__eq__close(self):
        tol = 0.8 * HTOL
        d4 = Domain([-2, 0, 1, 3, 5])
        d5 = Domain([-2 * (1 + tol), 0 - tol, 1 + tol, 3 * (1 + tol), 5 * (1 - tol)])
        d6 = Domain(
            [
                -2 * (1 + 2 * tol),
                0 - 2 * tol,
                1 + 2 * tol,
                3 * (1 + 2 * tol),
                5 * (1 - 2 * tol),
            ]
        )
        assert d4 == d5
        assert d4 != d6

    def test__ne__(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        assert not (d1 != d2)
        assert d1 != d3

    def test__ne___result_type(self):
        d1 = Domain([-2, 0, 1, 3, 5])
        d2 = Domain([-2, 0, 1, 3, 5])
        d3 = Domain([-1, 1])
        assert isinstance(d1 != d2, bool)
        assert isinstance(d1 != d3, bool)

    def test_from_chebfun(self):
        ff = chebfun(lambda x: np.cos(x), np.linspace(-10, 10, 11))
        Domain.from_chebfun(ff)

    def test_breakpoints_in(self):
        d1 = Domain([-1, 0, 1])
        d2 = Domain([-2, 0.5, 1, 3])

        result1 = d1.breakpoints_in(d2)
        assert isinstance(result1, np.ndarray)
        assert result1.size == 3
        assert not result1[0]
        assert not result1[1]
        assert result1[2]

        result2 = d2.breakpoints_in(d1)
        assert isinstance(result2, np.ndarray)
        assert result2.size == 4
        assert not result2[0]
        assert not result2[1]
        assert result2[2]
        assert not result2[3]

        assert d1.breakpoints_in(d1).all()
        assert d2.breakpoints_in(d2).all()
        assert not d1.breakpoints_in(Domain([-5, 5])).any()
        assert not d2.breakpoints_in(Domain([-5, 5])).any()

    def test_breakpoints_in_close(self):
        tol = 0.8 * HTOL
        d1 = Domain([-1, 0, 1])
        d2 = Domain([-2, 0 - tol, 1 + tol, 3])
        result = d1.breakpoints_in(d2)
        assert not result[0]
        assert result[1]
        assert result[2]

    def test_support(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain(np.linspace(-10, 10, 51))
        assert np.all(dom_a.support.view(np.ndarray) == [-2, 1])
        assert np.all(dom_b.support.view(np.ndarray) == [-2, 1])
        assert np.all(dom_c.support.view(np.ndarray) == [-10, 10])

    def test_size(self):
        dom_a = Domain([-2, 1])
        dom_b = Domain([-2, 0, 1])
        dom_c = Domain(np.linspace(-10, 10, 51))
        assert dom_a.size == 2
        assert dom_b.size == 3
        assert dom_c.size == 51

    def test_restrict(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        dom_c = Domain(np.linspace(-2, 1, 16))
        assert dom_a.restrict(dom_b) == Domain([-1.5, -1, -0.5, 0, 0.5])
        assert dom_a.restrict(dom_c) == dom_c
        assert dom_a.restrict(dom_a) == dom_a
        assert dom_b.restrict(dom_b) == dom_b
        assert dom_c.restrict(dom_c) == dom_c
        # tests to check if catch breakpoints that are different by eps
        # (linspace introduces these effects)
        dom_d = Domain(np.linspace(-0.4, 0.4, 2))
        assert dom_c.restrict(dom_d) == Domain([-0.4, -0.2, 0, 0.2, 0.4])

    def test_restrict_raises(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        dom_c = Domain(np.linspace(-2, 1, 16))
        with pytest.raises(NotSubdomain):
            dom_b.restrict(dom_a)
        with pytest.raises(NotSubdomain):
            dom_b.restrict(dom_c)

    def test_merge(self):
        dom_a = Domain([-2, -1, 0, 1])
        dom_b = Domain([-1.5, -0.5, 0.5])
        assert dom_b.merge(dom_a) == Domain([-2, -1.5, -1, -0.5, 0, 0.5, 1])

    def test_union(self):
        dom_a = Domain([-2, 0, 2])
        dom_b = Domain([-2, -1, 1, 2])
        assert dom_a.union(dom_b) != dom_a
        assert dom_a.union(dom_b) != dom_b
        assert dom_a.union(dom_b) == Domain([-2, -1, 0, 1, 2])
        assert dom_b.union(dom_a) == Domain([-2, -1, 0, 1, 2])

    def test_union_close(self):
        tol = 0.8 * HTOL
        dom_a = Domain([-2, 0, 2])
        dom_c = Domain([-2 - 2 * tol, -1 + tol, 1 + tol, 2 + 2 * tol])
        assert dom_a.union(dom_c) == Domain([-2, -1, 0, 1, 2])
        assert dom_c.union(dom_a) == Domain([-2, -1, 0, 1, 2])

    def test_union_raises(self):
        dom_a = Domain([-2, 0])
        dom_b = Domain([-2, 3])
        with pytest.raises(SupportMismatch):
            dom_a.union(dom_b)
        with pytest.raises(SupportMismatch):
            dom_b.union(dom_a)


@pytest.fixture
def check_funs_data():
    def f(x):
        return np.exp(x)

    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))
    fun2 = Bndfun.initfun_adaptive(f, Interval(-0.5, 0.5))
    fun3 = Bndfun.initfun_adaptive(f, Interval(2, 2.5))
    fun4 = Bndfun.initfun_adaptive(f, Interval(-3, -2))

    return {
        "fun0": fun0,
        "fun1": fun1,
        "fun2": fun2,
        "fun3": fun3,
        "fun4": fun4,
        "funs_a": np.array([fun1, fun0, fun2]),
        "funs_b": np.array([fun1, fun2]),
        "funs_c": np.array([fun0, fun3]),
        "funs_d": np.array([fun1, fun4])
    }

class TestCheckFuns:
    """Tests for the chebpy.core.utilities check_funs method"""

    def test_verify_empty(self):
        funs = check_funs(np.array([]))
        assert funs.size == 0

    def test_verify_contiguous(self, check_funs_data):
        fun0 = check_funs_data["fun0"]
        fun1 = check_funs_data["fun1"]
        funs = check_funs(np.array([fun0, fun1]))
        assert funs[0] == fun0
        assert funs[1] == fun1

    def test_verify_sort(self, check_funs_data):
        fun0 = check_funs_data["fun0"]
        fun1 = check_funs_data["fun1"]
        funs = check_funs(np.array([fun1, fun0]))
        assert funs[0] == fun0
        assert funs[1] == fun1

    def test_verify_overlapping(self, check_funs_data):
        funs_a = check_funs_data["funs_a"]
        funs_b = check_funs_data["funs_b"]
        with pytest.raises(IntervalOverlap):
            check_funs(funs_a)
        with pytest.raises(IntervalOverlap):
            check_funs(funs_b)

    def test_verify_gap(self, check_funs_data):
        funs_c = check_funs_data["funs_c"]
        funs_d = check_funs_data["funs_d"]
        with pytest.raises(IntervalGap):
            check_funs(funs_c)
        with pytest.raises(IntervalGap):
            check_funs(funs_d)


@pytest.fixture
def compute_breakdata_fixture():
    def f(x):
        return np.exp(x)

    fun0 = Bndfun.initfun_adaptive(f, Interval(-1, 0))
    fun1 = Bndfun.initfun_adaptive(f, Interval(0, 1))

    return {
        "fun0": fun0,
        "fun1": fun1
    }

# tests for the chebpy.core.utilities compute_breakdata function
class TestComputeBreakdata:
    def test_compute_breakdata_empty(self):
        breaks = compute_breakdata(np.array([]))
        # list(...) for Python 2/3 compatibility
        assert np.array(list(breaks.items())).size == 0

    def test_compute_breakdata_1(self, compute_breakdata_fixture):
        fun0 = compute_breakdata_fixture["fun0"]
        funs = np.array([fun0])
        breaks = compute_breakdata(funs)
        x, y = list(breaks.keys()), list(breaks.values())
        assert infnorm(x - np.array([-1, 0])) <= eps
        assert infnorm(y - np.array([np.exp(-1), np.exp(0)])) <= 2 * eps

    def test_compute_breakdata_2(self, compute_breakdata_fixture):
        fun0 = compute_breakdata_fixture["fun0"]
        fun1 = compute_breakdata_fixture["fun1"]
        funs = np.array([fun0, fun1])
        breaks = compute_breakdata(funs)
        x, y = list(breaks.keys()), list(breaks.values())
        assert infnorm(x - np.array([-1, 0, 1])) <= eps
        assert infnorm(y - np.array([np.exp(-1), np.exp(0), np.exp(1)])) <= 2 * eps


# reset the testsfun variable so it doesn't get picked up by nose
testfun = None
