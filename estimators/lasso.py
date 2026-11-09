import numpy as np
from sklearn.linear_model import lasso_path

from .wsindy import _ls


class Lasso:
    """beta-hat = argmin (1/2K)||Y - X beta||^2 + lam ||D beta||_1 with D = diag(||X_l||/sqrt(K)): the lasso on the
    columns scaled to ||X_l||^2/K = 1, which is LohWainwright(standardize=True) with Sigma_W = 0 and no l1 radius.
    It is solved by scikit-learn's coordinate descent, warm-started along the path, as the update (2.15) of LW does
    not converge here (X'X/K has condition number ~1e9). beta-hat is returned in the original units.

    The path has n_lambdas values from lam_max = ||Z'Y||_inf/K, the least lam with beta-hat = 0, down to ratio lam_max,
    log-spaced, for Z = X/D; lam-hat is chosen on it by
    select="mb":     the loss with which MSTLS chooses lambda (Messenger & Bortz 2021),
                     ||X(beta - beta_LS)||/||X beta_LS|| + ||beta||_0/P with beta_LS least squares on every column,
                     at its smallest minimizing lam as in WSINDy
    select="oracle": ||beta - beta_*||_2, with beta_* = truth.coefficients(equation) for a PDE truth. A reference only:
                     beta_* is unknown in practice
    tol is scikit-learn's tolerance on the duality gap, relative to ||Y||^2/K.
    """

    def __init__(self, select="mb", n_lambdas=50, ratio=1e-4, truth=None, tol=1e-8, max_iter=1000000):
        if select not in ("mb", "oracle"):
            raise ValueError('select must be "mb" or "oracle"')
        if select == "oracle" and truth is None:
            raise ValueError('select="oracle" needs the PDE as truth')
        self.select, self.truth = select, truth
        self.n_lambdas, self.ratio = int(n_lambdas), ratio
        self.tol, self.max_iter = tol, int(max_iter)

    def estimate(self, wf, equation):
        X, Y = wf.system(equation)
        betas = self.path(X, Y)
        if self.select == "mb":
            beta_ls = _ls(X, Y)
            loss = [np.linalg.norm(X@(b - beta_ls))/np.linalg.norm(X@beta_ls) + np.mean(b != 0) for b in betas]
        else:
            loss = [np.linalg.norm(b - self.truth.coefficients(equation)) for b in betas]
        # The path runs from the largest lam down, so the last minimizer has the smallest lam
        return betas[len(loss) - 1 - np.argmin(loss[::-1])]

    def lambdas(self, X, Y):
        Z = X/self._scale(X)
        return np.abs(Z.T@Y).max()/len(X)*np.geomspace(1, self.ratio, self.n_lambdas)

    def path(self, X, Y):
        # beta-hat in the original units at each lam of lambdas(X, Y), from the largest down, shape (n_lambdas, P)
        D = self._scale(X)
        _, coefs, _ = lasso_path(X/D, Y, alphas=self.lambdas(X, Y), tol=self.tol, max_iter=self.max_iter)
        return coefs.T/D

    @staticmethod
    def _scale(X):
        D = np.sqrt(np.mean(X**2, axis=0))
        return np.where(D > 0, D, 1.)
