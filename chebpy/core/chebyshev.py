"""Immutable representation of Chebyshev polynomials.

This module provides a dataclass for the immutable representation of Chebyshev
polynomials and various build functions to construct such polynomials.
"""

import dataclasses
import numpy as np

from .algorithms import (
    adaptive,
    chebpts2,
    vals2coeffs2,
)
from .utilities import Interval


@dataclasses.dataclass(frozen=True)
class ChebyshevPolynomial:
    """Immutable representation of a Chebyshev polynomial.

    This class represents a Chebyshev polynomial using its coefficients in the
    Chebyshev basis. The polynomial is defined on a specific interval.

    Attributes:
        coeffs: The coefficients of the Chebyshev polynomial.
        interval: The interval on which the polynomial is defined.
    """
    coeffs: np.ndarray
    interval: Interval

    def __post_init__(self):
        """Validate and normalize the inputs after initialization."""
        # Convert coeffs to numpy array if it's not already
        if not isinstance(self.coeffs, np.ndarray):
            object.__setattr__(self, 'coeffs', np.array(self.coeffs))

        # Convert interval to Interval if it's not already
        if not isinstance(self.interval, Interval):
            object.__setattr__(self, 'interval', Interval(*self.interval))

    def __call__(self, x):
        """Evaluate the polynomial at the given points.

        Args:
            x: Points at which to evaluate the polynomial.

        Returns:
            The values of the polynomial at the given points.
        """
        # Map x from the interval to [-1, 1]
        y = self.interval.invmap(x)

        # Evaluate using Clenshaw's algorithm
        return self._clenshaw(y)

    #@staticmethod
    def _clenshaw(self, x):
        """Evaluate a Chebyshev polynomial using Clenshaw's algorithm.

        Args:
            x: Points at which to evaluate the polynomial.
            coeffs: Coefficients of the Chebyshev polynomial.

        Returns:
            The values of the polynomial at the given points.
        """
        if len(self.coeffs) == 0:
            return np.zeros_like(x)

        x = np.asarray(x)
        bk1 = np.zeros_like(x)
        bk2 = np.zeros_like(x)

        for k in range(len(self.coeffs) - 1, 0, -1):
            bk = self.coeffs[k] + 2 * x * bk1 - bk2
            bk2 = bk1
            bk1 = bk

        return self.coeffs[0] + x * bk1 - bk2


def from_coefficients(coeffs, interval=None):
    """Create a Chebyshev polynomial from its coefficients.

    Args:
        coeffs: The coefficients of the Chebyshev polynomial.
        interval: The interval on which the polynomial is defined.
            Defaults to [-1, 1].

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial with the given coefficients.
    """
    interval = interval if interval is not None else [-1, 1]
    return ChebyshevPolynomial(coeffs, interval)


def from_function(func, interval=None, n=None):
    """Create a Chebyshev polynomial from a function.

    Args:
        func: The function to approximate.
        interval: The interval on which to approximate the function.
            Defaults to [-1, 1].
        n: The number of points to use for the approximation.
            If None, uses adaptive sampling.

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial approximating the function.
    """
    interval = interval if interval is not None else [-1, 1]
    interval_obj = Interval(*interval) if not isinstance(interval, Interval) else interval

    if n is None:
        # Use adaptive sampling
        coeffs = adaptive_sampling(func, interval_obj)
    else:
        # Use fixed sampling
        coeffs = fixed_sampling(func, interval_obj, n)

    return ChebyshevPolynomial(coeffs, interval_obj)


def from_values(values, interval=None):
    """Create a Chebyshev polynomial from values at Chebyshev points.

    Args:
        values: The values of the function at Chebyshev points.
        interval: The interval on which the polynomial is defined.
            Defaults to [-1, 1].

    Returns:
        ChebyshevPolynomial: A new Chebyshev polynomial with the given values.
    """
    interval = interval if interval is not None else [-1, 1]
    coeffs = vals2coeffs2(values)
    return ChebyshevPolynomial(coeffs, interval)


def adaptive_sampling(func, interval):
    """Sample a function adaptively to determine its Chebyshev coefficients.

    Args:
        func: The function to sample.
        interval: The interval on which to sample the function.

    Returns:
        numpy.ndarray: The Chebyshev coefficients of the function.
    """
    # Map the function to the standard domain [-1, 1]
    mapped_func = lambda y: func(interval(y))

    # Use the adaptive algorithm to determine the coefficients
    coeffs = adaptive(ChebyshevPolynomial, mapped_func, hscale=interval.hscale)

    return from_coefficients(coeffs, interval)



def fixed_sampling(func, interval, n):
    """Sample a function at a fixed number of points to determine its Chebyshev coefficients.

    Args:
        func: The function to sample.
        interval: The interval on which to sample the function.
        n: The number of points to use for the sampling.

    Returns:
        numpy.ndarray: The Chebyshev coefficients of the function.
    """
    # Map the function to the standard domain [-1, 1]
    mapped_func = lambda y: func(interval(y))

    # Sample the function at Chebyshev points
    points = chebpts2(n)
    from numpy.polynomial.chebyshev import chebpts2 as np_chebpts2
    points = np_chebpts2(n)
    values = mapped_func(points)

    # Convert the values to coefficients
    coeffs = vals2coeffs2(values)

    return from_coefficients(coeffs, interval)