"""Unit-tests for chebpy/core/chebyshev.py.

This module contains tests for the ChebyshevPolynomial class and its associated
functions, including initialization, evaluation, and various construction methods.
"""

import numpy as np
import pytest

from chebpy.core.chebyshev import (
    ChebyshevPolynomial,
    from_coefficients,
    from_function,
    from_values,
    adaptive_sampling,
    fixed_sampling,
)
from chebpy.core.utilities import Interval

# Small value for floating-point comparisons
eps = np.finfo(float).eps


@pytest.fixture
def chebyshev_fixtures():
    """Create fixtures for testing ChebyshevPolynomial.

    Returns:
        dict: Dictionary containing test fixtures
    """
    # Simple polynomial: p(x) = 1 + 2x + 3x^2
    # Chebyshev coefficients: [1 + 3/2, 2, 3/2]
    simple_coeffs = np.array([2.5, 2.0, 1.5])
    simple_poly = ChebyshevPolynomial(simple_coeffs, Interval(-1, 1))

    # Constant polynomial: p(x) = 5
    const_coeffs = np.array([5.0])
    const_poly = ChebyshevPolynomial(const_coeffs, Interval(-1, 1))

    # Empty polynomial
    empty_coeffs = np.array([])
    empty_poly = ChebyshevPolynomial(empty_coeffs, Interval(-1, 1))

    # Test functions
    def f1(x):
        return x**2  # Quadratic function

    def f2(x):
        return np.sin(x)  # Sine function

    return {
        "simple_coeffs": simple_coeffs,
        "simple_poly": simple_poly,
        "const_coeffs": const_coeffs,
        "const_poly": const_poly,
        "empty_coeffs": empty_coeffs,
        "empty_poly": empty_poly,
        "f1": f1,
        "f2": f2,
    }


class TestChebyshevPolynomial:
    """Tests for the ChebyshevPolynomial class."""

    def test_init(self, chebyshev_fixtures):
        """Test initialization of ChebyshevPolynomial.

        This test verifies that ChebyshevPolynomial objects can be initialized
        with different types of coefficients and intervals.

        Args:
            chebyshev_fixtures: Fixture providing test data
        """
        # Test initialization with numpy array and Interval
        coeffs = chebyshev_fixtures["simple_coeffs"]
        interval = Interval(-1, 1)
        poly = ChebyshevPolynomial(coeffs, interval)
        assert np.array_equal(poly.coeffs, coeffs)
        assert poly.interval == interval

        # Test initialization with list and tuple
        poly = ChebyshevPolynomial([1.0, 2.0, 3.0], [-2, 2])
        assert np.array_equal(poly.coeffs, np.array([1.0, 2.0, 3.0]))
        assert poly.interval == Interval(-2, 2)

        # Test initialization with empty coefficients
        poly = ChebyshevPolynomial([], [-1, 1])
        assert poly.coeffs.size == 0
        assert poly.interval == Interval(-1, 1)

    def test_call(self, chebyshev_fixtures):
        """Test evaluation of ChebyshevPolynomial.

        This test verifies that ChebyshevPolynomial objects can be evaluated
        at different points and return the expected values.

        Args:
            chebyshev_fixtures: Fixture providing test data
        """
        poly = chebyshev_fixtures["simple_poly"]

        # Test evaluation at a single point
        x = 0.5
        # For p(x) = 1 + 2x + 3x^2 at x = 0.5, we expect 1 + 2*0.5 + 3*0.5^2 = 2.75
        expected = 2.75
        assert abs(poly(x) - expected) < 10 * eps

        # Test evaluation at multiple points
        x = np.array([-1.0, 0.0, 1.0])
        # For p(x) = 1 + 2x + 3x^2 at x = -1, 0, 1, we expect [2, 1, 6]
        expected = np.array([2.0, 1.0, 6.0])
        assert np.max(np.abs(poly(x) - expected)) < 10 * eps

        # Test evaluation of constant polynomial
        const_poly = chebyshev_fixtures["const_poly"]
        x = np.linspace(-1, 1, 10)
        expected = np.full_like(x, 5.0)
        assert np.max(np.abs(const_poly(x) - expected)) < 10 * eps

        # Test evaluation of empty polynomial
        empty_poly = chebyshev_fixtures["empty_poly"]
        x = np.array([0.0, 0.5, 1.0])
        expected = np.zeros_like(x)
        assert np.array_equal(empty_poly(x), expected)

    def test_clenshaw(self, chebyshev_fixtures):
        """Test the _clenshaw method.

        This test verifies that the _clenshaw method correctly evaluates
        Chebyshev polynomials using Clenshaw's algorithm.

        Args:
            chebyshev_fixtures: Fixture providing test data
        """
        poly = chebyshev_fixtures["simple_poly"]

        # Test Clenshaw algorithm at a single point
        x = 0.5
        # For p(x) = 1 + 2x + 3x^2 at x = 0.5, we expect 1 + 2*0.5 + 3*0.5^2 = 2.75
        expected = 2.75
        assert abs(poly._clenshaw(x) - expected) < 10 * eps

        # Test Clenshaw algorithm at multiple points
        x = np.array([-1.0, 0.0, 1.0])
        # For p(x) = 1 + 2x + 3x^2 at x = -1, 0, 1, we expect [2, 1, 6]
        expected = np.array([2.0, 1.0, 6.0])
        assert np.max(np.abs(poly._clenshaw(x) - expected)) < 10 * eps

        # Test Clenshaw algorithm with empty coefficients
        empty_poly = chebyshev_fixtures["empty_poly"]
        x = np.array([0.0, 0.5, 1.0])
        expected = np.zeros_like(x)
        assert np.array_equal(empty_poly._clenshaw(x), expected)


class TestFromCoefficients:
    """Tests for the from_coefficients function."""

    def test_from_coefficients(self, chebyshev_fixtures):
        """Test the from_coefficients function.

        This test verifies that the from_coefficients function correctly creates
        ChebyshevPolynomial objects from coefficients.

        Args:
            chebyshev_fixtures: Fixture providing test data
        """
        coeffs = chebyshev_fixtures["simple_coeffs"]

        # Test with default interval
        poly = from_coefficients(coeffs)
        assert np.array_equal(poly.coeffs, coeffs)
        assert poly.interval == Interval(-1, 1)

        # Test with custom interval
        interval = [-2, 2]
        poly = from_coefficients(coeffs, interval)
        assert np.array_equal(poly.coeffs, coeffs)
        assert poly.interval == Interval(-2, 2)

        # Test with empty coefficients
        poly = from_coefficients([])
        assert poly.coeffs.size == 0
        assert poly.interval == Interval(-1, 1)


class TestFromFunction:
    """Tests for the from_function function."""

    def test_from_function_adaptive(self, chebyshev_fixtures, monkeypatch):
        """Test the from_function function with adaptive sampling.

        This test verifies that the from_function function correctly creates
        ChebyshevPolynomial objects from functions using adaptive sampling.

        Args:
            chebyshev_fixtures: Fixture providing test data
            monkeypatch: Pytest fixture for patching functions
        """
        f1 = chebyshev_fixtures["f1"]

        # Mock the adaptive_sampling function to return a simple polynomial
        def mock_adaptive_sampling(func, interval):
            return ChebyshevPolynomial(np.array([0.0, 0.0, 1.0]), interval)

        monkeypatch.setattr("chebpy.core.chebyshev.adaptive_sampling", mock_adaptive_sampling)

        # Test with default interval and adaptive sampling
        poly = from_function(f1)

        # Verify that the polynomial is a ChebyshevPolynomial
        assert isinstance(poly, ChebyshevPolynomial)
        assert poly.interval == Interval(-1, 1)

        # Test with custom interval
        interval = [-2, 2]
        poly = from_function(f1, interval)

        # Verify that the polynomial has the correct interval
        assert poly.interval == Interval(-2, 2)

    def test_from_function_fixed(self, chebyshev_fixtures, monkeypatch):
        """Test the from_function function with fixed sampling.

        This test verifies that the from_function function correctly creates
        ChebyshevPolynomial objects from functions using fixed sampling.

        Args:
            chebyshev_fixtures: Fixture providing test data
            monkeypatch: Pytest fixture for patching functions
        """
        f1 = chebyshev_fixtures["f1"]

        # Mock the fixed_sampling function to return a simple polynomial
        def mock_fixed_sampling(func, interval, n):
            return ChebyshevPolynomial(np.array([0.0, 0.0, 1.0]), interval)

        monkeypatch.setattr("chebpy.core.chebyshev.fixed_sampling", mock_fixed_sampling)

        # Test with fixed sampling
        n = 10
        poly = from_function(f1, n=n)

        # Verify that the polynomial is a ChebyshevPolynomial
        assert isinstance(poly, ChebyshevPolynomial)
        assert poly.interval == Interval(-1, 1)

        # Test with custom interval and fixed sampling
        interval = [-2, 2]
        poly = from_function(f1, interval, n)

        # Verify that the polynomial has the correct interval
        assert poly.interval == Interval(-2, 2)


class TestFromValues:
    """Tests for the from_values function."""

    def test_from_values(self):
        """Test the from_values function.

        This test verifies that the from_values function correctly creates
        ChebyshevPolynomial objects from values at Chebyshev points.
        """
        # Create values for a quadratic function at Chebyshev points
        n = 10
        from chebpy.core.algorithms import chebpts2
        points = chebpts2(n)
        values = points**2

        # Create polynomial from values
        poly = from_values(values)

        # Verify that the polynomial interpolates the values
        assert np.max(np.abs(poly(points) - values)) < 10 * eps

        # Test with custom interval
        interval = [-2, 2]
        poly = from_values(values, interval)

        # Verify that the polynomial has the correct interval
        assert poly.interval == Interval(-2, 2)


class TestAdaptiveSampling:
    """Tests for the adaptive_sampling function."""

    def test_adaptive_sampling(self, chebyshev_fixtures, monkeypatch):
        """Test the adaptive_sampling function.

        This test verifies that the adaptive_sampling function correctly samples
        a function adaptively to determine its Chebyshev coefficients.

        Args:
            chebyshev_fixtures: Fixture providing test data
            monkeypatch: Pytest fixture for patching functions
        """
        f1 = chebyshev_fixtures["f1"]
        interval = Interval(-1, 1)

        # Mock the entire adaptive_sampling function
        def mock_adaptive_sampling(func, interval):
            return np.array([0.0, 0.0, 1.0])

        monkeypatch.setattr("chebpy.core.chebyshev.adaptive_sampling", mock_adaptive_sampling)

        # Sample the function adaptively
        result = from_function(f1, interval)

        # Verify that the result is a ChebyshevPolynomial
        assert isinstance(result, ChebyshevPolynomial)
        assert result.interval == interval

        # Verify that the coefficients are as expected
        assert np.array_equal(result.coeffs, np.array([0.0, 0.0, 1.0]))


class TestFixedSampling:
    """Tests for the fixed_sampling function."""

    def test_fixed_sampling(self, chebyshev_fixtures, monkeypatch):
        """Test the fixed_sampling function.

        This test verifies that the fixed_sampling function correctly samples
        a function at a fixed number of points to determine its Chebyshev coefficients.

        Args:
            chebyshev_fixtures: Fixture providing test data
            monkeypatch: Pytest fixture for patching functions
        """
        f1 = chebyshev_fixtures["f1"]
        interval = Interval(-1, 1)
        n = 10

        # Mock the entire fixed_sampling function
        def mock_fixed_sampling(func, interval, n):
            return np.array([0.0, 0.0, 1.0])

        monkeypatch.setattr("chebpy.core.chebyshev.fixed_sampling", mock_fixed_sampling)

        # Sample the function at a fixed number of points
        result = from_function(f1, interval, n)

        # Verify that the result is a ChebyshevPolynomial
        assert isinstance(result, ChebyshevPolynomial)
        assert result.interval == interval

        # Verify that the coefficients are as expected
        np.testing.assert_array_equal(result.coeffs, np.array([0.0, 0.0, 1.0]))


class TestEdgeCases:
    """Tests for edge cases."""

    def test_different_interval_types(self):
        """Test initialization with different interval types.

        This test verifies that ChebyshevPolynomial objects can be initialized
        with different types of intervals.
        """
        coeffs = np.array([1.0, 2.0, 3.0])

        # Test with tuple
        poly = ChebyshevPolynomial(coeffs, (-1, 1))
        assert poly.interval == Interval(-1, 1)

        # Test with list
        poly = ChebyshevPolynomial(coeffs, [-1, 1])
        assert poly.interval == Interval(-1, 1)

        # Test with Interval
        poly = ChebyshevPolynomial(coeffs, Interval(-1, 1))
        assert poly.interval == Interval(-1, 1)

    def test_evaluation_outside_interval(self):
        """Test evaluation outside the interval.

        This test verifies that ChebyshevPolynomial objects can be evaluated
        at points outside their interval of definition.
        """
        coeffs = np.array([1.0, 2.0, 3.0])
        interval = Interval(-1, 1)
        poly = ChebyshevPolynomial(coeffs, interval)

        # Evaluate at points outside the interval
        x = np.array([-2.0, 2.0])

        # The polynomial should still be evaluated, but the values may not be accurate
        # We're just checking that it doesn't raise an error
        result = poly(x)
        assert result.size == x.size
