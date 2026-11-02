from dataclasses import dataclass
from itertools import product
from math import comb, factorial

import numpy as np


def _moment(j, x, s2):
    # E[(x + eps)^j] for eps ~ N(0, s2)
    return sum(comb(j, r)*factorial(r)//(2**(r//2)*factorial(r//2))*s2**(r//2)*x**(j - r) for r in range(0, j + 1, 2))


@dataclass(frozen=True)
class Term:
    """The monomial u_1^p_1 ... u_c^p_c, with the derivative alpha moved onto the test function.

    powers = (p_1, ..., p_c), derivative = alpha per coordinate. In (u, v) and (x, t),
    Term((0, 1), (2, 0)) is v_xx and Term((2, 1), (0, 0)) is u^2 v.
    """

    powers: tuple
    derivative: tuple

    def __call__(self, u):
        return np.prod([u[c]**p for c, p in enumerate(self.powers)], axis=0)

    def covariance(self, other, u, s2):
        # Cov(f(u + eps), g(u + eps)) pointwise, for independent eps_c ~ N(0, s2_c)
        joint, means = 1., 1.
        for c, (a, b) in enumerate(zip(self.powers, other.powers)):
            joint = joint*_moment(a + b, u[c], s2[c])
            means = means*_moment(a, u[c], s2[c])*_moment(b, u[c], s2[c])
        return joint - means


@dataclass(frozen=True)
class Equation:
    """lhs = sum_l beta_l rhs_l."""

    lhs: Term
    rhs: tuple


def library(alphas, J, c):
    # The constant, which no derivative can act on, then every monomial of total degree j = 1, ..., J in c components
    # under each alpha^s, ordered by s, then j
    powers = sorted((p for p in product(range(J + 1), repeat=c) if 1 <= sum(p) <= J), key=sum)
    return (Term((0,)*c, (0,)*len(alphas[0])), *(Term(p, alpha) for alpha in alphas for p in powers))
