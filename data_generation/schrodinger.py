import numpy as np

from weak_form import Equation, Term

from .base import PDE

D_XX, D_T, D_NONE = (2, 0), (0, 1), (0, 0)


class Schrodinger(PDE):
    """i psi_t + a psi_xx + b |psi|^2 psi = 0 with a, b > 0, written with psi = u - iv as

    u_t =  a v_xx + b u^2 v + b v^3,
    v_t = -a u_xx - b u^3 - b u v^2.

    u_* is the rescaled Satsuma-Yajima solution, so psi(x, 0) = 2/sqrt(b) sech(x/sqrt(2a)).
    """

    reference_shape = (1024, 1001)

    def __init__(self, a=0.5, b=1.0, domain=((-5.0, 5.0), (0.0, np.pi))):
        if a <= 0 or b <= 0:
            raise ValueError("the closed-form solution requires a, b > 0")
        self.a, self.b, self.domain = a, b, domain
        self.equations = [
            Equation(Term((1, 0), D_T), (Term((0, 1), D_XX), Term((2, 1), D_NONE), Term((0, 3), D_NONE))),
            Equation(Term((0, 1), D_T), (Term((1, 0), D_XX), Term((3, 0), D_NONE), Term((1, 2), D_NONE))),
        ]
        self.beta_star = [np.array([a, b, b]), np.array([-a, -b, -b])]

    def solution(self, x, t):
        # (u, v) = (Re psi, -Im psi)
        s = x/np.sqrt(2*self.a)
        psi = 4*np.exp(.5j*t)*(np.cosh(3*s) + 3*np.exp(4j*t)*np.cosh(s)) / (np.cosh(4*s) + 4*np.cosh(2*s) + 3*np.cos(4*t))
        return np.stack([psi.real, -psi.imag])/np.sqrt(self.b)
