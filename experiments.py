"""Experiment settings. Run one with: python main.py <name>

pde:            PDE class with its coefficients
sampling, size: "grid" with the grid shape, or "random" with n
rho:            noise level, sigma_c = rho * RMS(u_*c)
iterations:     Monte Carlo replications; replication i uses seed i
estimator:      any object with estimate(wf, equation), or a dict of them by name, all run on the same weak forms and
                saved as results/<setting>_<name>.npz
dictionary:     candidate terms per equation, in the order of pde.equations
test_functions: the test-function family, or a callable that builds it from the data
pilot:          optional, pilot(data, seed) -> u-hat from pilot.py, fit to all n observations. X_n and Y_n are then
                computed from u-hat at new points instead of from U
points, m:      with a pilot, where the new points are: "random" for m uniform points summed by Monte Carlo (m = n by
                default), or "grid" for the grid of shape m summed by the rectangle rule (the data's grid by default)
folds, debias:  with a pilot, optional cross-fitting (weak_form.CrossFitWeakForm): u-hat_f is fit outside fold f, and
                X_n, Y_n average its terms at the m new points, with the first-order correction of L^db on fold f when
                debias (pde_discovery.pdf, Sec. 5)
cross_fit:      with folds, False for the sample splitting of Algorithm 2 in place of cross-fitting: u-hat from the odd
                observations only, the correction from the even ones, and no average over folds
"""
from functools import partial

import numpy as np

from data_generation import Schrodinger
from estimators import BestSubset, CoCoLasso, Lasso, LohWainwright, WSINDy
from pilot import PINN, perturbed_truth
from weak_form import Equation, TestFunctions, Term, library, spectral_test_functions

# Grid designs, with the settings of nls_debiasing.ipynb: K = 40 x 35 = 1400
R = (30*10/256, 40*np.pi/250)
NLS_PHI = TestFunctions((-5 + R[0] + np.arange(40)*5*10/256, R[1] + np.arange(35)*5*np.pi/250), R, (11, 10))

NLS_EQUATIONS = Schrodinger().equations
# True terms plus the alias column: v in u_t, u in v_t
NLS_ALIAS = [Equation(eq.lhs, eq.rhs + (Term(alias, (0, 0)),)) for eq, alias in zip(NLS_EQUATIONS, ((0, 1), (1, 0)))]
# The constant, and d_x^s for s = 0, ..., S on the monomials of degree j = 1, ..., J in (u, v), S = J = 6: 1 + 7 x 27 = 190 columns
NLS_LIBRARY = library(tuple((s, 0) for s in range(7)), 6, 2)
NLS_FULL = [Equation(eq.lhs, NLS_LIBRARY) for eq in NLS_EQUATIONS]
# Candidates of lam for LW and CoCoLasso, searched by the corrected CV (5.4)
NLS_LAMBDAS = np.geomspace(0.005, 1, 18)
# Oracle radius of (2.7) at rho = 0.1 with the spectral test functions: median over 20 seeds and both equations of sqrt(k) ||D beta_*||_2
NLS_LW_RADIUS = {1000: 1.63, 5000: 1.03, 10000: 0.91}
# The Schrodinger example of Messenger & Bortz (2021), Tables 3 and 4, on its grid of 256 x 251, at every n:
# m = (19, 25), s = (5, 5), p = (11, 10), K = 44 x 41 = 1804; the oracle radius is computed as above
R_WSINDY_PDE = (19*10/256, 25*np.pi/250)
NLS_WSINDY_PDE = TestFunctions((-5 + R_WSINDY_PDE[0] + np.arange(44)*5*10/256, R_WSINDY_PDE[1] + np.arange(41)*5*np.pi/250), R_WSINDY_PDE, (11, 10))
NLS_WSINDY_PDE_RADIUS = {1000: 0.79, 5000: 0.47, 10000: 0.43}
# The same rule at rho = 0.1, 0.5, 1, keyed by (n, rho); rho = 0.1 repeats NLS_WSINDY_PDE_RADIUS
NLS_WSINDY_PDE_RADII = {
    (1000, 0.1): 0.79, (1000, 0.5): 0.87, (1000, 1.0): 1.09,
    (5000, 0.1): 0.47, (5000, 0.5): 0.51, (5000, 1.0): 0.62,
    (10000, 0.1): 0.43, (10000, 0.5): 0.46, (10000, 1.0): 0.53,
}
# u-hat = u_* + zeta n^(-1/3) g with zeta = 1, n the sample size and g a random cosine series of unit RMS
NLS_PILOT = partial(perturbed_truth, pde=Schrodinger(a=0.5, b=1.0))
# u-hat = a network fit to the n observations, with the number of epochs (at most 5000) chosen on 20% held out,
# then refit on all n
NLS_PINN = PINN(validation=0.2, refit=True)
# MSTLS, and the lasso with lambda chosen by the MSTLS loss or, as a reference, by the distance to beta_*
NLS_SELECTORS = {"mstls": WSINDy("mstls"), "lasso": Lasso("mb"), "lasso_oracle": Lasso("oracle", truth=Schrodinger(a=0.5, b=1.0))}
# Best subset selection (Bertsimas, King & Mazumder 2016) on k = 1, ..., 10, with one path per system shared by both;
# k-hat by the MSTLS loss or, as a reference, by the distance to beta_*
_BEST_SUBSET_PATHS = {}
NLS_BEST_SUBSET = {
    "mb": BestSubset("mb", cache=_BEST_SUBSET_PATHS),
    "oracle": BestSubset("oracle", truth=Schrodinger(a=0.5, b=1.0), cache=_BEST_SUBSET_PATHS),
}
NLS_ESTIMATORS = {
    "mstls": lambda radius: WSINDy("mstls"),
    "lw": lambda radius: LohWainwright(NLS_LAMBDAS, radius, standardize=True),
    "coco": lambda radius: CoCoLasso(NLS_LAMBDAS),
}

EXPERIMENTS = {
    "nls_grid_fixed": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="grid", size=(256, 251), rho=0.1, iterations=20,
        estimator=WSINDy("fixed"), dictionary=NLS_ALIAS, test_functions=NLS_PHI,
    ),
    "nls_random_fixed": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=64256, rho=0.1, iterations=20,
        estimator=WSINDy("fixed"), dictionary=NLS_ALIAS, test_functions=partial(spectral_test_functions, equations=NLS_ALIAS),
    ),
    "nls_grid_mstls": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="grid", size=(256, 251), rho=0.1, iterations=20,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=NLS_PHI,
    ),
    "nls_random_mstls": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=64256, rho=0.1, iterations=20,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=partial(spectral_test_functions, equations=NLS_FULL),
    ),
    **{
        f"nls_random_n{n}_{name}": dict(
            pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=n, rho=0.1, iterations=100,
            estimator=estimator(NLS_LW_RADIUS[n]), dictionary=NLS_FULL, test_functions=partial(spectral_test_functions, equations=NLS_FULL),
        )
        for n in NLS_LW_RADIUS for name, estimator in NLS_ESTIMATORS.items()
    },
    **{
        f"nls_random_n{n}_{name}_wsindy_pde": dict(
            pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=n, rho=0.1, iterations=100,
            estimator=estimator(NLS_WSINDY_PDE_RADIUS[n]), dictionary=NLS_FULL, test_functions=NLS_WSINDY_PDE,
        )
        for n in NLS_WSINDY_PDE_RADIUS for name, estimator in NLS_ESTIMATORS.items()
    },
    # u-hat from all n points, evaluated on the 256 x 251 grid of Messenger & Bortz (2021)
    "nls_random_n1000_mstls_wsindy_pde_uhat_grid": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=1000, rho=0.1, iterations=100,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=NLS_WSINDY_PDE, pilot=NLS_PILOT,
        points="grid", m=(256, 251),
    ),
    "nls_random_n1000_mstls_wsindy_pde_pinn_grid": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=1000, rho=0.1, iterations=100,
        estimator=WSINDy("mstls"), dictionary=NLS_FULL, test_functions=NLS_WSINDY_PDE, pilot=NLS_PINN,
        points="grid", m=(256, 251),
    ),
    # The lasso on the same u-hat as nls_random_n1000_mstls_wsindy_pde_pinn_grid
    "nls_random_n1000_wsindy_pde_pinn_grid": dict(
        pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=1000, rho=0.1, iterations=100,
        estimator={k: NLS_SELECTORS[k] for k in ("lasso", "lasso_oracle")}, dictionary=NLS_FULL,
        test_functions=NLS_WSINDY_PDE, pilot=NLS_PINN, points="grid", m=(256, 251),
    ),
    # Cross-fitting over the odd and even indices: u-hat_f from the other half on the grid, without and with L^db
    **{
        f"nls_random_n1000_wsindy_pde_pinn_crossfit{suffix}": dict(
            pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=1000, rho=0.1, iterations=100,
            estimator=NLS_SELECTORS, dictionary=NLS_FULL, test_functions=NLS_WSINDY_PDE, pilot=NLS_PINN,
            points="grid", m=(256, 251), folds=2, debias=debias,
        )
        for suffix, debias in (("", False), ("_db", True))
    },
    # Best subset selection on 10 seeds: L(beta; U) from the data, L(beta; u-hat) with u-hat on all n, and L(beta; u-hat)
    # without and with L^db for u-hat cross-fit or, as in Algorithm 2, from one split
    **{
        f"nls_random_n1000_wsindy_pde_{variant}_best_subset": dict(
            pde=Schrodinger(a=0.5, b=1.0), sampling="random", size=1000, rho=0.1, iterations=10,
            estimator=NLS_BEST_SUBSET, dictionary=NLS_FULL, test_functions=NLS_WSINDY_PDE, **extra,
        )
        for variant, extra in (
            ("data", {}),
            ("pinn_grid", dict(pilot=NLS_PINN, points="grid", m=(256, 251))),
            ("pinn_crossfit", dict(pilot=NLS_PINN, points="grid", m=(256, 251), folds=2, debias=False)),
            ("pinn_crossfit_db", dict(pilot=NLS_PINN, points="grid", m=(256, 251), folds=2, debias=True)),
            ("pinn_split", dict(pilot=NLS_PINN, points="grid", m=(256, 251), folds=2, debias=False, cross_fit=False)),
            ("pinn_split_db", dict(pilot=NLS_PINN, points="grid", m=(256, 251), folds=2, debias=True, cross_fit=False)),
        )
    },
}
