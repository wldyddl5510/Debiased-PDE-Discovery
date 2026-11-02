import numpy as np

from weak_form import WeakForm

from .solvers import _composite_gradient, _nearest_psd


class LohWainwright:
    """Loh & Wainwright (2012), eq. (2.7) with (2.8), taking W = X_n(U) - X(u_*) as the additive noise:

    beta-hat in argmin_{||beta||_1 <= radius} beta'Gamma beta/2 - gamma'beta + lam ||beta||_1,
    Gamma = X'X/K - Sigma_W,  gamma = X'Y/K,  radius = b_0 sqrt(k).

    Solved by the update (2.15) from beta = 0, with eta the largest absolute eigenvalue of Gamma.

    standardize=True (not in the paper) solves this on the columns scaled to ||X_l||^2/K = 1, so that lam and radius
    apply to D beta with D = diag(||X_l||/sqrt(K)), and returns beta in the original units.

    lam is a number, or a grid searched (not in the paper) by the corrected V-fold CV (5.4) of Datta & Zou (2017) over
    the z_i of a random design.
    """

    project = False

    def __init__(self, lam, radius, standardize=False, folds=5, tol=1e-10, max_iter=1000000):
        self.lam = lam
        self.radius = radius
        self.standardize = standardize
        self.folds = folds
        self.tol = tol
        self.max_iter = max_iter

    def estimate(self, wf, equation):
        X = wf.system(equation)[0]
        D = np.sqrt(np.mean(X**2, axis=0)) if self.standardize else np.ones(X.shape[1])
        lam = self._select(wf, equation, D) if np.ndim(self.lam) else self.lam
        return self._solve(self._moments(wf, equation, D, self.project), lam)/D

    def _moments(self, wf, equation, D, project):
        # (Gamma, gamma) on the columns X/D, with Gamma replaced by (Gamma)_+ if project
        X, Y = wf.system(equation)
        Z = X/D
        Gamma = Z.T@Z/len(X) - wf.noise_covariance(equation)/np.outer(D, D)
        return _nearest_psd(Gamma) if project else Gamma, Z.T@Y/len(X)

    def _solve(self, moments, lam):
        return _composite_gradient(*moments, lam, self.radius, self.tol, self.max_iter)

    def _select(self, wf, equation, D):
        # The corrected CV (5.4): beta-hat without fold v, scored with (Gamma_v)_+ and gamma_v of fold v
        n = wf.data.U.shape[-1]
        loss = np.zeros(len(self.lam))
        for fold in np.array_split(np.arange(n), self.folds):
            train = self._moments(WeakForm(wf.data._select(np.setdiff1d(np.arange(n), fold)), wf.phi), equation, D, self.project)
            Gamma, gamma = self._moments(WeakForm(wf.data._select(fold), wf.phi), equation, D, True)
            for j, lam in enumerate(self.lam):
                beta = self._solve(train, lam)
                loss[j] += beta@Gamma@beta - 2*gamma@beta
        return self.lam[np.argmin(loss)]
