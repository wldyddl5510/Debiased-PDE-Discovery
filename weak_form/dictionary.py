from dataclasses import dataclass

import numpy as np


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


@dataclass(frozen=True)
class Equation:
    """lhs = sum_l beta_l rhs_l."""

    lhs: Term
    rhs: tuple
