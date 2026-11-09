from dataclasses import dataclass, replace
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
    """X_n(u) and Y_n(u) for one data set and one family of test functions, or a callable that builds it from the data.

    By default u = U. A callable u(*z), such as a pilot u-hat, replaces the data by u at new points z'_j on the domain:
    points="random": m points drawn uniformly (m = n by default), summed by Monte Carlo with weight |Omega|/m;
    points="grid":   the grid of shape m (the data's grid by default), half-open in space and closed in time like
                     PDE.grid, summed by the rectangle rule with the cell volume, as in WSINDy_PDE's convolutions.
    The test functions are built from the observed data in both cases.
    """

    def __init__(self, data, test_functions, u=None, points="random", m=None, seed=None):
        self.phi = test_functions(data) if callable(test_functions) else test_functions
        if u is not None:
            if points == "random":
                rng = np.random.default_rng(seed)
                z = tuple(rng.uniform(lo, hi, data.U[0].size if m is None else m) for lo, hi in data.domain)
                data = replace(data, coords=z, design="random")
            elif points == "grid":
                if m is None and data.design != "grid":
                    raise ValueError('points="grid" needs the grid shape m for a random design')
                shape, last = data.U.shape[1:] if m is None else m, len(data.domain) - 1
                axes = tuple(np.linspace(lo, hi, N, endpoint=a == last) for a, ((lo, hi), N) in enumerate(zip(data.domain, shape)))
                data = replace(data, coords=axes, design="grid")
            else:
                raise ValueError('points must be "random" or "grid"')
            # u carries no observation noise: Sigma_W is 0 on a grid and the Monte Carlo covariance at random points
            data = replace(data, U=u(*data.points), u_star=None, sigma=np.zeros(len(data.U)))
        self.data = data
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


class CrossFitWeakForm:
    """X_n and Y_n from u-hats fit by cross-fitting, with the first-order correction of L^db (pde_discovery.pdf, Sec. 5,
    Algorithm 2) or without it.

    The observations, along the last axis on a grid, are split into the folds I_f = {f, f + folds, f + 2 folds, ...}:
    odd and even indices for two folds, as in Algorithm 2. For each f, u-hat_f = pilot(the data outside I_f), and
        X_f = X_m(u-hat_f) + dz_f sum_{i in I_f} phi~_kl(z_i) grad f_l(u-hat_f(z_i)) . (U_i - u-hat_f(z_i)),
        Y_f = Y_m(u-hat_f) + dz_f sum_{i in I_f} phi~_0k(z_i) grad f_0(u-hat_f(z_i)) . (U_i - u-hat_f(z_i)),
    with X_m, Y_m the sums of u-hat_f at the m new points of WeakForm (points, m), and the correction the Gateaux
    derivative of X, Y at u-hat_f in the direction U - u-hat_f, a Monte Carlo sum over I_f with dz_f = |Omega|/|I_f|
    on a random design. U_i - u-hat_f(z_i) is unbiased for u_*(z_i) - u-hat_f(z_i) since u-hat_f never saw I_f.
    X_n, Y_n are the means of X_f, Y_f over f; debias=False leaves the correction out. score() gives instead the
    debiased score (b^db, H^db) of Sec. 5.1 from the same fits.

    cross_fit=False is the sample splitting of Algorithm 2 instead: only the last fold is held out, so with two folds
    u-hat is fit to the odd observations i = 1, 3, ... (I_0, counting from 1 as the PDF does), the correction is summed
    over the even ones (I_1), and X_n, Y_n are those of that one fold. Its u-hat is the cross-fit's u-hat of that fold.

    seed is an integer or None: fold f fits its pilot with seed (seed, 1, f) and draws its new points with (seed, 2, f).
    """

    def __init__(self, data, test_functions, pilot, points="grid", m=None, folds=2, debias=True, seed=None, cross_fit=True):
        if int(folds) != folds or folds < 2:
            raise ValueError("folds must be an integer >= 2")
        self.phi = test_functions(data) if callable(test_functions) else test_functions
        self.data, self.debias = data, bool(debias)
        key = (lambda *k: None) if seed is None else (lambda *k: (seed, *k))
        N = data.U.shape[-1]
        # Per fold: the weak form of u-hat_f at the new points, that of the held-out data, u-hat_f there, and U - u-hat_f
        self._parts = []
        for f in range(folds) if cross_fit else (folds - 1,):
            held = np.arange(f, N, folds)
            uhat = pilot(data._select(np.setdiff1d(np.arange(N), held)), seed=key(1, f))
            out = WeakForm(data._select(held), self.phi)
            values = uhat(*out.data.points)
            self._parts.append((WeakForm(data, self.phi, uhat, points, m, seed=key(2, f)), out, values, out.data.U - values))
        self._systems = {}

    def system(self, equation):
        # (X_n, Y_n), shapes (K, P) and (K,)
        parts = self._folds(equation)
        if not self.debias:
            return np.mean([X for X, Y, C, c in parts], axis=0), np.mean([Y for X, Y, C, c in parts], axis=0)
        return np.mean([X + C for X, Y, C, c in parts], axis=0), np.mean([Y + c for X, Y, C, c in parts], axis=0)

    def score(self, equation):
        # (b^db, H^db) of pde_discovery.pdf, Sec. 5.1, the means over the folds of
        #     b_f = (X_f'Y_f + C_f'Y_f + X_f'c_f)/K,   H_f = (X_f'X_f + C_f'X_f + X_f'C_f)/K,
        # with X_f, Y_f the terms of u-hat_f at the new points and C_f, c_f their correction on fold f: the score
        # X'(Y - X beta)/K expanded to first order in U - u-hat_f, so that the correction enters only linearly and its
        # noise has mean 0. Fold by fold, as C_f is independent of u-hat_f but not of the other u-hats. debias is ignored
        parts = self._folds(equation)
        K = len(parts[0][0])
        b = np.mean([X.T@Y + C.T@Y + X.T@c for X, Y, C, c in parts], axis=0)/K
        H = np.mean([X.T@X + C.T@X + X.T@C for X, Y, C, c in parts], axis=0)/K
        return b, H

    def _folds(self, equation):
        # Per fold (X_f, Y_f, C_f, c_f): the terms of u-hat_f at the new points and their correction on fold f,
        # computed once per equation
        if equation not in self._systems:
            parts = []
            for new, out, values, residual in self._parts:
                X, Y = new.system(equation)
                # d/dt f(u-hat + t r) at t = 0 is grad f(u-hat) . r, summed over the components
                direction = lambda term: out.integrate(np.sum(term.gradient(values)*residual, axis=0), term.derivative)
                parts.append((X, Y, np.stack([direction(term) for term in equation.rhs], axis=1), direction(equation.lhs)))
            self._systems[equation] = parts
        return self._systems[equation]


class SplitWeakForm:
    """X_n and Y_n by sample splitting: u-hat is fit to one half of the data, and the first-order correction of L^db
    (pde_discovery.pdf, Sec. 5) is computed on the other half, which u-hat has not seen. It is one fold of
    CrossFitWeakForm, without the swap of the halves and the average:
        X_n = X_m(u-hat) + dz' sum_{i in held} phi~_kl(z_i) grad f_l(u-hat(z_i)) . (U_i - u-hat(z_i)),
        Y_n = Y_m(u-hat) + dz' sum_{i in held} phi~_0k(z_i) grad f_0(u-hat(z_i)) . (U_i - u-hat(z_i)),
    with X_m, Y_m the sums of u-hat at the m new points of WeakForm (points, m), and dz' = |Omega|/|held| on a random
    design. debias=False leaves the correction out. score() gives the debiased score (b^db, H^db) of Sec. 5.1.

    uhat is already fit, to the observations outside held; held is a Data of the other observations, such as the
    second element of data.split(). The test functions are built from all of data, as in CrossFitWeakForm, and seed
    draws the new points when points="random".
    """

    def __init__(self, data, test_functions, uhat, held, points="grid", m=None, debias=True, seed=None):
        self.phi = test_functions(data) if callable(test_functions) else test_functions
        self.data, self.debias = data, bool(debias)
        # The weak form of u-hat at the new points, that of the held-out data, u-hat there, and U - u-hat
        self._new = WeakForm(data, self.phi, uhat, points, m, seed=seed)
        self._held = WeakForm(held, self.phi)
        self._values = uhat(*self._held.data.points)
        self._residual = self._held.data.U - self._values
        self._systems = {}

    def system(self, equation):
        # (X_n, Y_n), shapes (K, P) and (K,)
        X, Y, C, c = self._parts(equation)
        return (X + C, Y + c) if self.debias else (X, Y)

    def score(self, equation):
        # (b^db, H^db) of pde_discovery.pdf, Sec. 5.1 for the one split, as in CrossFitWeakForm.score:
        #     b = (X'Y + C'Y + X'c)/K,   H = (X'X + C'X + X'C)/K. debias is ignored
        X, Y, C, c = self._parts(equation)
        K = len(X)
        return (X.T@Y + C.T@Y + X.T@c)/K, (X.T@X + C.T@X + X.T@C)/K

    def _parts(self, equation):
        # (X, Y, C, c): the terms of u-hat at the new points and their correction on held, computed once per equation
        if equation not in self._systems:
            X, Y = self._new.system(equation)
            # d/dt f(u-hat + t r) at t = 0 is grad f(u-hat) . r, summed over the components
            direction = lambda term: self._held.integrate(np.sum(term.gradient(self._values)*self._residual, axis=0),
                                                          term.derivative)
            self._systems[equation] = X, Y, np.stack([direction(term) for term in equation.rhs], axis=1), direction(equation.lhs)
        return self._systems[equation]
