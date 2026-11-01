import numpy as np

LAMBDAS = 10.0**(-4 + 4*np.arange(50)/49)


def _ls(X, Y):
    return np.linalg.lstsq(X, Y, rcond=None)[0]


def _threshold(X, Y, lam, beta):
    # Iterated thresholding at a fixed lambda, starting from the least-squares beta
    ratio = np.linalg.norm(Y)/np.linalg.norm(X, axis=0)
    lower, upper = lam*np.maximum(1, ratio), np.minimum(1, ratio)/lam
    support = np.ones(X.shape[1], bool)
    for _ in range(X.shape[1]):
        keep = (lower <= np.abs(beta)) & (np.abs(beta) <= upper)
        if np.array_equal(keep, support):
            break
        support, beta = keep, np.zeros_like(beta)
        beta[support] = _ls(X[:, support], Y)
    return beta


class WSINDy:
    """beta-hat in Y_n(u) = X_n(u) beta, for one equation.

    support="fixed": least squares on every column of X.
    support="mstls": MSTLS of Messenger & Bortz (2021) on X and Y as given, without their rescaling.
                     Its loss charges 1/P per kept column, so it drops true terms when P is small.
    """

    def __init__(self, support="fixed", lambdas=LAMBDAS):
        if support not in ("fixed", "mstls"):
            raise ValueError('support must be "fixed" or "mstls"')
        self.support = support
        self.lambdas = np.sort(lambdas)

    def fit(self, X, Y):
        beta_ls = _ls(X, Y)
        if self.support == "fixed":
            return beta_ls
        fits = [_threshold(X, Y, lam, beta_ls) for lam in self.lambdas]
        loss = [np.linalg.norm(X@(b - beta_ls))/np.linalg.norm(X@beta_ls) + np.mean(b != 0) for b in fits]
        return fits[np.argmin(loss)]  # argmin takes the first, so the smallest minimizing lambda
