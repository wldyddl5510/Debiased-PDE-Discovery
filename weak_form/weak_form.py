from dataclasses import dataclass
from itertools import combinations_with_replacement

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
    """X_n(u) and Y_n(u) for one data set and one family of test functions, or a callable that builds it from the data."""

    def __init__(self, data, test_functions):
        self.data = data
        self.phi = test_functions(data) if callable(test_functions) else test_functions
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

    def noise_covariance(self, equation):
        # Sigma-hat_W = (1/K) sum_k Cov(W_k) for W = X_n(U) - X(u_*), with U_i in place of u_*(z_i).
        # Grid: only the noise is random, so use the closed-form Cov(f_l(u + eps), f_l'(u + eps)) at U_i.
        # Random: the summands are i.i.d. over i, so use their sample covariance, which also covers the z_i.
        data, terms = self.data, equation.rhs
        grid = data.design == "grid"
        if grid and data.sigma is None:
            raise ValueError("a grid design needs Data.sigma")
        d = len(data.coords)
        S = np.zeros((len(terms), len(terms)))
        if grid:
            for l, m in combinations_with_replacement(range(len(terms)), 2):
                f, g = terms[l], terms[m]
                G = [np.sum(self._basis(a, f.derivative[a])*self._basis(a, g.derivative[a]), axis=0) for a in range(d)]
                factors = [x for a in range(d) for x in (G[a], [a])]
                S[l, m] = S[m, l] = np.einsum(*factors, f.covariance(g, data.U, data.sigma**2), list(range(d)), [], optimize=True)
        else:
            # Terms under the same derivative pair share their weight at each z_i, so fill S block by block
            F = np.stack([term(data.U) for term in terms], axis=1)
            blocks = {}
            for l, term in enumerate(terms):
                blocks.setdefault(term.derivative, []).append(l)
            for (alpha, ls), (gamma, ms) in combinations_with_replacement(blocks.items(), 2):
                w = np.prod([np.sum(self._basis(a, alpha[a])*self._basis(a, gamma[a]), axis=0) for a in range(d)], axis=0)
                S[np.ix_(ls, ms)] = (F[:, ls]*w[:, None]).T@F[:, ms]
                S[np.ix_(ms, ls)] = S[np.ix_(ls, ms)].T
        K = np.prod([len(c) for c in self.phi.centers])
        S = S*data.dz**2/K
        if not grid:
            X = self.system(equation)[0]
            S = S - X.T@X/(K*data.U.shape[-1])
        return S

    def _basis(self, axis, order):
        if (axis, order) not in self._bases:
            self._bases[axis, order] = self.phi.basis(axis, self.data.coords[axis], order)
        return self._bases[axis, order]
