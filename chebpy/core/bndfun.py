"""Implementation of functions on bounded intervals.

This module provides the Bndfun class, which extends Classicfun to represent
functions on bounded intervals [a,b]. It is a concrete implementation of the
abstract Classicfun class, specifically designed for bounded domains.
"""

from .classicfun import Classicfun


class Bndfun(Classicfun):
    """Class to approximate functions on bounded intervals [a,b]."""

    @classmethod
    def initfun(cls, f, interval=None, n=None):
        """Initialize a Bndfun from a callable function.

        This constructor automatically selects between the adaptive or fixed-length
        constructor based on the input arguments passed.

        Args:
            f (callable): The function to be approximated.
            interval (array-like, optional): The interval on which to define the function.
                Defaults to None, which uses the default interval from preferences.
            n (int, optional): The number of degrees of freedom to use. If None,
                uses adaptive construction. Defaults to None.

        Returns:
            Bndfun: A new instance representing the function.
        """
        if n is None:
            return cls.initfun_adaptive(f, interval)
        else:
            return cls.initfun_fixedlen(f, interval, n)

    @classmethod
    def initfun_adaptive(cls, f, interval):
        """Initialize a Bndfun from a callable function using adaptive sampling.

        This constructor determines the appropriate number of points needed to
        represent the function to the specified tolerance using an adaptive algorithm.

        Args:
            f (callable): The function to be approximated.
            interval (array-like): The interval on which to define the function.

        Returns:
            Bndfun: A new instance representing the function.
        """
        return super().initfun_adaptive(f, interval)

    @classmethod
    def initfun_fixedlen(cls, f, interval, n):
        """Initialize a Bndfun from a callable function using a fixed number of points.

        This constructor creates a function representation using exactly n points.

        Args:
            f (callable): The function to be approximated.
            interval (array-like): The interval on which to define the function.
            n (int): The number of points to use.

        Returns:
            Bndfun: A new instance representing the function.
        """
        return super().initfun_fixedlen(f, interval, n)

    @classmethod
    def initvalues(cls, values, interval=None):
        """Initialize a Bndfun from values at Chebyshev points.

        This constructor creates a function representation from values at
        Chebyshev points on the specified interval.

        Args:
            values (array-like): Values at Chebyshev points.
            interval (array-like, optional): The interval on which to define the function.
                Defaults to None, which uses the default interval from preferences.

        Returns:
            Bndfun: A new instance representing the function.
        """
        from .chebtech import Chebtech
        onefun = Chebtech.initvalues(values, interval=interval)
        return cls(onefun, interval)

    def prolong(self, n):
        """Extend the function representation to a larger size.

        This method extends the function representation to use more coefficients
        or a higher degree, which can be useful for certain operations.

        Args:
            n (int): The new size for the function representation.

        Returns:
            Bndfun: A new function with an extended representation.
        """
        onefun = self.onefun.prolong(n)
        return self.__class__(onefun, self.interval)
