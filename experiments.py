"""Experiment settings. Run one with: python main.py <name>

pde:            PDE class with its coefficients
sampling, size: "grid" with the grid shape, or "random" with n
rho:            noise level, sigma_c = rho * RMS(u_*c)
iterations:     Monte Carlo replications; replication i uses seed i
estimator:      any object with fit(X, Y)
dictionary:     candidate terms per equation, in the order of pde.equations
test_functions: the test-function family
"""
import numpy as np

from data_generation import Schrodinger
from estimators import WSINDy
from weak_form import Equation, TestFunctions, Term

# Settings of nls_debiasing.ipynb: K = 40 x 35 = 1400
R = (30*10/256, 40*np.pi/250)
NLS_PHI = TestFunctions((-5 + R[0] + np.arange(40)*5*10/256, R[1] + np.arange(35)*5*np.pi/250), R, (11, 10))

NLS_EQUATIONS = Schrodinger().equations
# True terms plus the alias column: v in u_t, u in v_t
NLS_ALIAS = [Equation(eq.lhs, eq.rhs + (Term(alias, (0, 0)),)) for eq, alias in zip(NLS_EQUATIONS, ((0, 1), (1, 0)))]
# Every monomial of degree 1 to 3 in (u, v), plus the first and second x derivatives of u and v
NLS_LIBRARY = tuple(Term((i, j), (0, 0)) for i in range(4) for j in range(4) if 0 < i + j <= 3) \
    + tuple(Term(p, d) for p in ((1, 0), (0, 1)) for d in ((1, 0), (2, 0)))
NLS_FULL = [Equation(eq.lhs, NLS_LIBRARY) for eq in NLS_EQUATIONS]

EXPERIMENTS = {
    "nls_grid_fixed": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="grid", size=(256, 251), rho=0.1, iterations=20,
        estimator=WSINDy("fixed"), dictionary=NLS_ALIAS, test_functions=NLS_PHI,
    ),
    "nls_random_fixed": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=64256, rho=0.1, iterations=20,
        estimator=WSINDy("fixed"), dictionary=NLS_ALIAS, test_functions=NLS_PHI,
    ),
    "nls_grid_mstls": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="grid", size=(256, 251), rho=0.1, iterations=20,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=NLS_PHI,
    ),
    "nls_random_mstls": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=64256, rho=0.1, iterations=20,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=NLS_PHI,
    ),
}
