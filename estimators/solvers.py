import warnings

import numpy as np


def _soft_threshold(beta, t):
    return np.sign(beta)*np.maximum(np.abs(beta) - t, 0)


def _project_l1(beta, radius):
    # Duchi et al. (2008)
    if np.abs(beta).sum() <= radius:
        return beta
    s = np.sort(np.abs(beta))[::-1]
    excess = (np.cumsum(s) - radius)/np.arange(1, len(s) + 1)
    return _soft_threshold(beta, excess[s > excess][-1])


def _composite_gradient(Gamma, gamma, lam, radius, tol, max_iter):
    # The update (2.15) of Loh & Wainwright (2012) from beta = 0, with eta the largest absolute eigenvalue of Gamma
    step = 1/np.abs(np.linalg.eigvalsh(Gamma)).max()
    beta = np.zeros_like(gamma)
    for _ in range(max_iter):
        update = _project_l1(_soft_threshold(beta - step*(Gamma@beta - gamma), step*lam), radius)
        if np.linalg.norm(update - beta) <= tol*max(1, np.linalg.norm(beta)):
            return update
        beta = update
    warnings.warn("the update (2.15) did not converge")
    return beta


def _nearest_psd(S, mu=10.0, eps=1e-3, tol=1e-5, max_iter=100000):
    # argmin_{A >= eps I} ||A - S||_max by the ADMM of Datta & Zou (2017), Algorithm 1.
    # The B step is the exact prox of ||.||_max: over all entries, with the l1 radius mu/2 of (A.6)
    B, Lam = np.zeros_like(S), np.zeros_like(S)
    for _ in range(max_iter):
        # ADMM part
        w, v = np.linalg.eigh(B + S + mu*Lam)
        A = (v*np.maximum(w, eps))@v.T
        C = A - S - mu*Lam
        previous, B = B, C - _project_l1(C.ravel(), mu/2).reshape(C.shape)
        Lam = Lam - (A - B - S)/mu
        if max(np.abs(A - B - S).max(), np.abs(B - previous).max()) <= tol:
            return A
    warnings.warn("the ADMM for the nearest PSD matrix did not converge")
    return A
