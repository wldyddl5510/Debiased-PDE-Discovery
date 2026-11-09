import hashlib

import numpy as np

from .wsindy import _ls


class BestSubset:
    """Best subset selection of Bertsimas, King & Mazumder (2016), for k = 1, ..., k_max:

        beta-hat_k in argmin ||Y - X beta||_2^2  s.t.  ||beta||_0 <= k,

    with (X, Y) from wf.system, so it runs on L(beta; U), L(beta; u-hat) or L^db (pde_discovery.pdf, Algorithm 2) alike.
    beta-hat_k does not depend on the scale of the columns, which are scaled to ||X_l||^2/K = 1, and Y to ||Y||^2/K = 1,
    for the numerics only; the columns are not centered, as the constant is a column of the dictionary.

    For each k, as in the paper:
    1. Algorithm 2 (hard thresholding of a gradient step 1/L, L = lambda_max(X'X), then a line search), from the starts
       of Section 5.2.1 (0 and starts - 1 draws of N(0, 4I)) and from beta-hat_(k-1), each polished by least squares on
       its support; the best is the warm start.
    2. The MIO (9), with SOS-1 constraints (beta_l, 1 - z_l), sum z_l <= k, and the bounds of Section 2.3.3:
       ||beta||_inf <= M_U = tau ||beta-warm||_inf, ||beta||_1 <= k M_U. Its quadratic is written as ||zeta||^2/2 with
       zeta = R beta - Q'Y from the QR factorization X = QR, which is (9) up to a constant and convex by construction.
       It is solved by Gurobi within time_limit seconds. If a bound binds, M_U is doubled and the MIO solved again.
    3. beta is refit by least squares on its support, unless a bound still binds.

    k-hat is chosen on the path by
    select="mb":     the MSTLS loss (Messenger & Bortz 2021), ||X(beta - beta_LS)||/||X beta_LS|| + ||beta||_0/P with
                     beta_LS least squares on every column, at its smallest minimizing k
    select="oracle": ||beta - beta_*||_2, with beta_* = truth.coefficients(equation). A reference only
    select=<int>:    that k

    The path is cached on (X, Y), so estimators that share a cache solve it once; path(X, Y) also returns, for each k,
    the solver's record: status, MIP gap, run time, M_U, whether a bound binds, and the objective ||Pi(Y - X beta)||^2/
    (2||Y||^2) of the warm start and of beta-hat_k, with Pi the projection on the span of X (the part that depends on beta).
    """

    def __init__(self, select="mb", k_max=10, truth=None, starts=50, tau=2.0, time_limit=30.0, threads=0, tol=1e-4,
                 max_iter=2000, seed=0, cache=None):
        if select not in ("mb", "oracle") and not isinstance(select, int):
            raise ValueError('select must be "mb", "oracle" or an int')
        if select == "oracle" and truth is None:
            raise ValueError('select="oracle" needs the PDE as truth')
        if isinstance(select, int) and not 1 <= select <= k_max:
            raise ValueError("select must be in 1, ..., k_max")
        self.select, self.truth, self.k_max = select, truth, int(k_max)
        self.starts, self.tau, self.time_limit, self.threads = int(starts), tau, time_limit, int(threads)
        self.tol, self.max_iter, self.seed = tol, int(max_iter), seed
        self._cache = {} if cache is None else cache
        self._env = None

    def estimate(self, wf, equation):
        X, Y = wf.system(equation)
        betas, _ = self.path(X, Y)
        return betas[self.choose(X, Y, betas, equation) - 1]

    def choose(self, X, Y, betas, equation):
        # k-hat in 1, ..., k_max
        if isinstance(self.select, int):
            return self.select
        if self.select == "mb":
            beta_ls = _ls(X, Y)
            loss = [np.linalg.norm(X@(b - beta_ls))/np.linalg.norm(X@beta_ls) + np.mean(b != 0) for b in betas]
        else:
            loss = [np.linalg.norm(b - self.truth.coefficients(equation)) for b in betas]
        return int(np.argmin(loss)) + 1

    def path(self, X, Y):
        # (beta-hat_k in the original units for k = 1, ..., k_max, shape (k_max, P); the solver's record per k)
        key = (hashlib.sha1(np.ascontiguousarray(X).tobytes() + np.ascontiguousarray(Y).tobytes()).hexdigest(),
               self.k_max, self.starts, self.tau, self.time_limit, self.tol, self.max_iter, self.seed)
        if key not in self._cache:
            self._cache[key] = self._path(X, Y)
        return self._cache[key]

    def _path(self, X, Y):
        K, P = X.shape
        D = np.sqrt(np.mean(X**2, axis=0))
        D = np.where(D > 0, D, 1.)
        s = np.sqrt(np.mean(Y**2))
        # ||Y - X beta||^2/(K s^2) = ||y - R b||^2 + const, with b = D beta/s
        Qz, R = np.linalg.qr(X/D/np.sqrt(K))
        y = Qz.T@(Y/s/np.sqrt(K))
        G, r = R.T@R, R.T@y
        L = np.linalg.eigvalsh(G)[-1]
        self._R, self._y = R, y
        rng = np.random.default_rng(self.seed)
        draws = [np.zeros(P)] + [rng.normal(0, 2, P) for _ in range(self.starts - 1)]
        betas, records, previous = np.zeros((self.k_max, P)), [], None
        for k in range(1, self.k_max + 1):
            warm = min((self._first_order(G, r, L, k, b0) for b0 in draws + ([previous] if previous is not None else [])),
                       key=lambda b: self._objective(R, y, b))
            b, record = self._mio(R, y, k, warm)
            record["warm"] = self._objective(R, y, warm)
            record["objective"] = self._objective(R, y, b)
            betas[k - 1], previous = b*s/D, b
            records.append(record)
        return betas, records

    @staticmethod
    def _objective(R, y, b):
        # ||y - R b||^2/2, which is ||Y - X beta||^2/(2 K s^2) up to the part of Y outside the span of X
        return 0.5*np.sum((y - R@b)**2)

    @staticmethod
    def _keep(c, k):
        # H_k(c): the k entries of largest magnitude
        out = np.zeros_like(c)
        top = np.argpartition(-np.abs(c), k - 1)[:k]
        out[top] = c[top]
        return out

    def _polish(self, support):
        # Least squares on the support, from R rather than the Gram matrix, whose condition number is its square
        b = np.zeros(self._R.shape[1])
        if support.any():
            b[support] = np.linalg.lstsq(self._R[:, support], self._y, rcond=None)[0]
        return b

    def _first_order(self, G, r, L, k, b):
        # Algorithm 2 from b, then least squares on the support of the last k-sparse iterate
        eta = self._keep(b, k)
        for _ in range(self.max_iter):
            grad = G@b - r
            eta = self._keep(b - grad/L, k)
            d = eta - b
            curvature = d@G@d
            step = -(grad@d)/curvature if curvature > 0 else 1.
            b_new = b + step*d
            if np.linalg.norm(b_new - b) <= self.tol:
                break
            b = b_new
        return self._polish(eta != 0)

    def _mio(self, R, y, k, warm):
        import gurobipy as gp
        from gurobipy import GRB

        if self._env is None:
            self._env = gp.Env(params={"OutputFlag": 0})
        P = len(warm)
        M = self.tau*max(np.abs(warm).max(), 1e-12)
        start = warm
        for attempt in range(4):
            with gp.Model(env=self._env) as model:
                model.Params.TimeLimit, model.Params.Threads, model.Params.Seed = self.time_limit, self.threads, 0
                beta = model.addMVar(P, lb=-M, ub=M)
                z = model.addMVar(P, vtype=GRB.BINARY)
                off = model.addMVar(P, lb=0, ub=1)  # 1 - z
                size = model.addMVar(P, lb=0, ub=M)  # |beta|
                zeta = model.addMVar(P, lb=-GRB.INFINITY)
                model.addConstr(off + z == 1)
                for l in range(P):
                    model.addSOS(GRB.SOS_TYPE1, [beta[l].item(), off[l].item()])
                model.addConstr(z.sum() <= k)
                model.addConstr(size >= beta)
                model.addConstr(size >= -beta)
                model.addConstr(size.sum() <= k*M)
                model.addConstr(R@beta - zeta == y)
                model.setObjective(0.5*(zeta@zeta))
                beta.Start, z.Start = start, (start != 0).astype(float)
                model.optimize()
                found = model.SolCount > 0
                b = np.where(z.X > 0.5, beta.X, 0.) if found else start
                record = dict(status=model.Status, gap=model.MIPGap if found else np.inf, time=model.Runtime, M_U=M)
            binds = np.abs(b).max() >= (1 - 1e-6)*M
            if not binds:
                break
            start, M = b, 2*M
        record["binds"] = bool(binds)
        b = b if binds else self._polish(b != 0)
        return b, record
