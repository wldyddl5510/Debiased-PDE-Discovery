from dataclasses import dataclass

import numpy as np
from numpy.polynomial import Polynomial


@dataclass(frozen=True, eq=False)
class TestFunctions:
    """phi_k(z) = prod_a (1 - ((z_a - c_a)/R_a)^2)^p_a over the tensor product of centers per coordinate."""

    centers: tuple
    radii: tuple
    degrees: tuple

    def basis(self, axis, z, order):
        # (-1)^order d^order/dz^order of the 1-D factors, shape (K_axis, len(z))
        b = (Polynomial([1, 0, -1])**self.degrees[axis]).deriv(order)
        R = self.radii[axis]
        r = (z - np.asarray(self.centers[axis])[:, None])/R
        return np.where(np.abs(r) < 1, (-1)**order*b(r)/R**order, 0.)


class WeakForm:
    """X_n(u) and Y_n(u) for one data set and one family of test functions."""

    def __init__(self, data, test_functions):
        self.data = data
        self.phi = test_functions
        self._bases = {}

    def integrate(self, F, order):
        # dz sum_i phi~_k(z_i) F(z_i) for every k, with order = alpha on phi per coordinate
        d = len(order)
        # Each factor of phi sums over its own axis on a grid, and over the shared points otherwise
        index = [d + a for a in range(d)] if self.data.design == "grid" else [d]*d
        factors = [x for a, m in enumerate(order) for x in (self._basis(a, m), [a, index[a]])]
        F_index = index if self.data.design == "grid" else [d]
        return self.data.dz*np.einsum(*factors, F, F_index, list(range(d)), optimize=True).ravel()

    def system(self, equation, u=None):
        # (X_n(u), Y_n(u)), shapes (K, P) and (K,); u is None for U, values at the z_i, or a callable u(*z)
        u = self.data.U if u is None else u(*self.data.points) if callable(u) else np.asarray(u)
        X = np.stack([self.integrate(term(u), term.derivative) for term in equation.rhs], axis=1)
        return X, self.integrate(equation.lhs(u), equation.lhs.derivative)

    def _basis(self, axis, order):
        if (axis, order) not in self._bases:
            self._bases[axis, order] = self.phi.basis(axis, self.data.coords[axis], order)
        return self._bases[axis, order]
