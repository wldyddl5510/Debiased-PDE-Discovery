from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np


@dataclass(frozen=True, eq=False)
class Data:
    """Observations (z_i, U_i), with time as the last coordinate.

    design="grid":   coords has one uniform axis per coordinate, U has shape (c, *axis lengths).
    design="random": coords has n values per coordinate, U has shape (c, n), and the z_i are
                     uniform on domain = ((lo, hi), ...).
    u_star holds the noiseless values u_*(z_i) and sigma the noise sd per component, when they are known.
    """

    coords: tuple
    U: np.ndarray
    design: str = "grid"
    domain: tuple | None = None
    u_star: np.ndarray | None = None
    sigma: np.ndarray | None = None

    def __post_init__(self):
        if self.design == "grid":
            if not all(np.allclose(np.diff(c), c[1] - c[0]) for c in self.coords):
                raise ValueError("grid axes must be uniform")
        elif self.design != "random" or self.domain is None:
            raise ValueError('design must be "grid", or "random" with a domain')

    @property
    def dz(self):
        # Quadrature weight: the grid cell volume, or |Omega|/n for random points
        if self.design == "grid":
            return np.prod([c[1] - c[0] for c in self.coords])
        return np.prod([hi - lo for lo, hi in self.domain])/self.U.shape[-1]

    @property
    def points(self):
        # The z_i, one array per coordinate
        return np.meshgrid(*self.coords, indexing="ij") if self.design == "grid" else self.coords

    @property
    def rms(self):
        return np.sqrt(np.mean(self.u_star**2, axis=tuple(range(1, self.u_star.ndim))))

    def add_noise(self, rho, seed=None, rms=None):
        # U_i = u_*(z_i) + eps_i with sd(eps_c) = rho * RMS(u_*c); RMS over these z_i by default
        sigma = rho*np.asarray(self.rms if rms is None else rms)
        noise = np.random.default_rng(seed).standard_normal(self.u_star.shape)
        return replace(self, U=self.u_star + sigma.reshape((-1,) + (1,)*(self.u_star.ndim - 1))*noise, sigma=sigma)

    def split(self):
        # (pilot, regression): odd/even time slices on a grid, second/first half of random points
        if self.design == "grid":
            return self._select(slice(1, None, 2)), self._select(slice(0, None, 2))
        mid = self.U.shape[-1]//2
        return self._select(slice(mid, None)), self._select(slice(mid))

    def _select(self, s):
        coords = (*self.coords[:-1], self.coords[-1][s]) if self.design == "grid" else tuple(c[s] for c in self.coords)
        u_star = None if self.u_star is None else self.u_star[..., s]
        return replace(self, coords=coords, U=self.U[..., s], u_star=u_star)


class PDE(ABC):
    """A PDE with known solution u_*.

    Subclasses set domain, reference_shape (the grid fixing the RMS), equations and beta_star.
    """

    domain: tuple
    reference_shape: tuple
    equations: list
    beta_star: list

    @abstractmethod
    def solution(self, *z):
        """u_*(z), shape (c, *z[0].shape)."""

    def coefficients(self, equation):
        # beta_* on the columns of equation, from the true equation with the same left-hand side; 0 for the other terms
        true = next(q for q, eq in enumerate(self.equations) if eq.lhs == equation.lhs)
        known = dict(zip(self.equations[true].rhs, self.beta_star[true]))
        return np.array([known.get(term, 0.) for term in equation.rhs])

    @cached_property
    def rms(self):
        # On a fixed grid, so that sigma_c is the same across draws and designs
        return self._clean_grid(self.reference_shape).rms

    def grid(self, shape, rho=0.0, seed=None):
        return self._clean_grid(shape).add_noise(rho, seed, self.rms)

    def random(self, n, rho=0.0, seed=None):
        rng = np.random.default_rng(seed)
        z = tuple(rng.uniform(lo, hi, n) for lo, hi in self.domain)
        u = self.solution(*z)
        return Data(z, u, "random", self.domain, u).add_noise(rho, rng, self.rms)

    def _clean_grid(self, shape):
        # Space axes are half-open, the time axis is closed
        last = len(shape) - 1
        axes = tuple(np.linspace(lo, hi, N, endpoint=a == last) for a, ((lo, hi), N) in enumerate(zip(self.domain, shape)))
        u = self.solution(*np.meshgrid(*axes, indexing="ij"))
        return Data(axes, u, "grid", self.domain, u)
