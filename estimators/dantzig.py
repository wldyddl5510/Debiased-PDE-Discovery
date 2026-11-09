import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


class DiscreteDantzig:
    """The discrete Dantzig selector (Mazumder & Radchenko 2017; pde_discovery.pdf, (9) and (10)):

        beta-hat in argmin ||beta||_0  s.t.  ||b - H beta||_inf <= lam,

    on the columns scaled by D = diag(||X_l||/sqrt(K)), with X from wf.system: b~ = b/D, H~ = H/(D D'), beta~ = D beta.
    score="db" takes (b, H) = (b^db, H^db) from wf.score, which needs a CrossFitWeakForm; score="plain" takes
    (X'Y/K, X'X/K) from wf.system. lam is a fraction of lam_max = ||b~||_inf, at which beta = 0 is feasible.

    Solved as a MILP by HiGHS (scipy.optimize.milp) with |beta~_l| <= big_m a_l, a_l in {0, 1}, in place of the SOS1
    constraint of the paper, and big_m a multiple of lam_max. Among the solutions of least support S the MILP returns
    any, so beta~_S is then refit on the score equations of S, H~_SS beta~_S = b~_S.
    """

    def __init__(self, lam, score="db", big_m=20.0, time_limit=120.0):
        if score not in ("db", "plain"):
            raise ValueError('score must be "db" or "plain"')
        self.lam, self.score, self.big_m, self.time_limit = lam, score, big_m, time_limit

    def estimate(self, wf, equation):
        X, Y = wf.system(equation)
        b, H = wf.score(equation) if self.score == "db" else (X.T@Y/len(X), X.T@X/len(X))
        return self.fit(b, H, np.sqrt(np.mean(X**2, axis=0)))[0]

    def fit(self, b, H, D):
        # (beta-hat in the original units, the MILP result)
        bt, Ht = b/D, H/np.outer(D, D)
        P, lam_max = len(b), np.abs(bt).max()
        lam, M = self.lam*lam_max, self.big_m*lam_max
        eye, zero = np.eye(P), np.zeros((P, P))
        constraints = [LinearConstraint(np.hstack([Ht, zero]), bt - lam, bt + lam),
                       LinearConstraint(np.block([[eye, -M*eye], [-eye, -M*eye]]), -np.inf, 0)]
        res = milp(np.r_[np.zeros(P), np.ones(P)], constraints=constraints, integrality=np.r_[np.zeros(P), np.ones(P)],
                   bounds=Bounds(np.r_[-M*np.ones(P), np.zeros(P)], np.r_[M*np.ones(P), np.ones(P)]),
                   options={"time_limit": self.time_limit})
        beta = np.zeros(P)
        if res.x is None:
            return beta, res
        support = res.x[P:] > 0.5
        if support.any():
            beta[support] = np.linalg.lstsq(Ht[np.ix_(support, support)], bt[support], rcond=None)[0]
        return beta/D, res
