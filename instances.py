"""The 9 instances of single_instance.ipynb, repeated over seeds for n = 1000, 5000, 10000 and rho = 0.1, 0.5, 1.

python instances.py job <n> <rho> <seed> [--threads T]
    One seed of one setting, as in the notebook, saved as results/instances/n<n>_rho<rho>_seed<seed>.npz
python instances.py run [--hours H] [--workers W] [--threads T]
    Every setting for seeds 0, 1, ..., 99, seed s of all 9 settings before seed s + 1, W jobs at a time with T threads
    each, until H hours have passed. Jobs still running then are stopped and not saved; saved seeds are skipped, so a
    second run continues the first.

Each file holds, for equation q = 0, 1:
    beta_<instance>_<q>         beta-hat of each instance in INSTANCES, and beta_star_<q>
    irr_<input>_<q>             (eta, mu) of the irrepresentable condition (Zhao & Yu 2006) for each Gram matrix in GRAMS
    dist_<input>_<q>            ||A - A(u_*)||/||A(u_*)|| for A = X on all terms, X on the true terms, and Y
    bss_gap_<input>_<q>         MIP gap of best subset at k = 1, ..., 10
    time_<instance>             seconds, for both equations
and epochs_<pilot>, l2_<pilot> (relative L^2(Omega) error of (u-hat, v-hat)) and time_<pilot> for both PINN fits.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SIZES = (1000, 5000, 10000)
RHOS = (0.1, 0.5, 1.0)
SEEDS = 100
OUT = Path("results")/"instances"
GRID = (256, 251)
# key: (name in the notebook, input, estimator)
INSTANCES = {
    "wsindy": ("WSINDy", "U", "mstls"),
    "lasso": ("Lasso", "U", "lasso"),
    "lw": ("Loh-Wainwright", "U", "lw"),
    "coco": ("CoCoLasso", "U", "coco"),
    "bss": ("Best-subset WSINDy", "U", "bss"),
    "plugin_wsindy": ("plug-in WSINDy", "plugin", "mstls"),
    "plugin_bss": ("plug-in best subset WSINDy", "plugin", "bss"),
    "db_wsindy": ("debiased WSINDy", "db", "mstls"),
    "db_bss": ("debiased best subset WSINDy", "db", "bss"),
}
INPUTS = {"U": "U_i", "plugin": "u-hat (all n)", "db": "u-hat (split), debiased"}
# Gram matrices of the irrepresentable condition: (input, minus Sigma_W)
GRAMS = {"truth": ("truth", False), "U": ("U", False), "U_eiv": ("U", True), "plugin": ("plugin", False), "db": ("db", False)}


def path(n, rho, seed):
    return OUT/f"n{n}_rho{rho:g}_seed{seed}.npz"


def irrepresentability(C, star):
    # (eta, mu): ||C_{S^c S} C_{SS}^{-1} sign(beta_*S)||_inf and its sign-free version ||C_{S^c S} C_{SS}^{-1}||_inf
    S = star != 0
    A = C[np.ix_(~S, S)]@np.linalg.inv(C[np.ix_(S, S)])
    return np.abs(A@np.sign(star[S])).max(), np.abs(A).sum(axis=1).max()


def gram(X, Sigma=0):
    # D^{-1} (X'X/K - Sigma) D^{-1}: the Gram matrix of the standardized columns, less Sigma_W for LW and CoCoLasso
    d = np.sqrt(np.mean(X**2, axis=0))
    return (X.T@X/len(X) - Sigma)/np.outer(d, d)


def estimators(n, rho, threads):
    from estimators import BestSubset, CoCoLasso, Lasso, LohWainwright, WSINDy
    from experiments import NLS_LAMBDAS, NLS_WSINDY_PDE_RADII

    return {
        "mstls": WSINDy("mstls"),
        "lasso": Lasso("mb"),
        "lw": LohWainwright(NLS_LAMBDAS, NLS_WSINDY_PDE_RADII[n, rho], standardize=True),
        "coco": CoCoLasso(NLS_LAMBDAS),
        "bss": BestSubset("mb", threads=threads),
    }


def run_seed(n, rho, seed, threads=1, pinn=None, methods=None):
    # The notebook for one seed; pinn and methods replace NLS_PINN and estimators(n, rho, threads), for tests
    from data_generation import Schrodinger
    from experiments import NLS_FULL, NLS_PINN, NLS_WSINDY_PDE
    from weak_form import SplitWeakForm, WeakForm

    pde = Schrodinger(a=0.5, b=1.0)
    pinn = NLS_PINN if pinn is None else pinn
    methods = estimators(n, rho, threads) if methods is None else methods
    dictionary, phi = NLS_FULL, NLS_WSINDY_PDE
    stars = [pde.coefficients(equation) for equation in dictionary]
    out = {f"beta_star_{q}": star for q, star in enumerate(stars)}

    # Data, and u-hat fit to all n observations or to one half, the other half kept for the correction
    data = pde.random(n, rho, seed)
    fit_half, debias_half = data.split()
    pilots = {}
    for name, observations, pilot_seed in (("all", data, (seed, 1)), ("split", fit_half, (seed, 1, 0))):
        start = time.time()
        pilots[name] = pinn(observations, seed=pilot_seed)
        out[f"time_{name}"], out[f"epochs_{name}"] = time.time() - start, pilots[name].epochs

    # Relative L^2(Omega) error of u-hat by the trapezoidal rule on the grid
    x, t = (np.linspace(*pde.domain[a], GRID[a]) for a in range(2))
    X_grid, T_grid = np.meshgrid(x, t, indexing="ij")
    u_star = pde.solution(X_grid, T_grid)
    l2 = lambda g: np.sqrt(np.trapezoid(np.trapezoid(g**2, t, axis=-1), x, axis=-1))
    for name, uhat in pilots.items():
        out[f"l2_{name}"] = l2(uhat(X_grid, T_grid) - u_star)/l2(u_star)

    weak_forms = {
        "U": WeakForm(data, phi),
        "plugin": WeakForm(data, phi, pilots["all"], "grid", GRID),
        "db": SplitWeakForm(data, phi, pilots["split"], debias_half, "grid", GRID, debias=True),
    }
    truth = WeakForm(data, phi, pde.solution, "grid", GRID)
    rel = lambda A, B: np.linalg.norm(A - B)/np.linalg.norm(B)
    for q, (equation, star) in enumerate(zip(dictionary, stars)):
        Xt, Yt = truth.system(equation)
        systems = {"truth": Xt, **{key: wf.system(equation)[0] for key, wf in weak_forms.items()}}
        for key, wf in weak_forms.items():
            X, Y = wf.system(equation)
            out[f"dist_{key}_{q}"] = np.array([rel(X, Xt), rel(X[:, star != 0], Xt[:, star != 0]), rel(Y, Yt)])
        Sigma = weak_forms["U"].noise_covariance(equation)
        for label, (key, eiv) in GRAMS.items():
            out[f"irr_{label}_{q}"] = np.array(irrepresentability(gram(systems[key], Sigma if eiv else 0), star))

    for key, (_, source, method) in INSTANCES.items():
        start = time.time()
        for q, equation in enumerate(dictionary):
            out[f"beta_{key}_{q}"] = methods[method].estimate(weak_forms[source], equation)
            if method == "bss":
                _, records = methods["bss"].path(*weak_forms[source].system(equation))
                out[f"bss_gap_{source}_{q}"] = np.array([r["gap"] for r in records])
        out[f"time_{key}"] = time.time() - start
    return out


def job(n, rho, seed, threads):
    import torch

    torch.set_num_threads(threads)
    start = time.time()
    result = run_seed(n, rho, seed, threads)
    target = path(n, rho, seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Written to a temporary file first, so that a stopped job leaves no file
    temporary = target.with_suffix(".tmp")
    with open(temporary, "wb") as f:
        np.savez(f, **result)
    os.replace(temporary, target)
    print(f"n={n} rho={rho:g} seed={seed}: saved {target} in {time.time() - start:.0f} s", flush=True)


def run(hours, workers, threads):
    deadline = time.time() + 3600*hours
    todo = [(n, rho, s) for s in range(SEEDS) for n in reversed(SIZES) for rho in RHOS if not path(n, rho, s).exists()]
    logs = OUT/"logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **{v: str(threads) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                      "VECLIB_MAXIMUM_THREADS")}}
    running, done, failed = {}, 0, []
    print(f"{len(todo)} jobs to run, {workers} at a time, until {time.ctime(deadline)}", flush=True)
    while todo or running:
        while todo and len(running) < workers and time.time() < deadline:
            n, rho, s = todo.pop(0)
            log = open(logs/f"n{n}_rho{rho:g}_seed{s}.txt", "w")
            command = [sys.executable, __file__, "job", str(n), str(rho), str(s), "--threads", str(threads)]
            running[subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)] = (n, rho, s, log, time.time())
        if time.time() >= deadline:
            for process, (n, rho, s, log, _) in running.items():
                process.terminate()
                process.wait()
                log.close()
                print(f"stopped n={n} rho={rho:g} seed={s} at the time limit", flush=True)
            break
        time.sleep(5)
        for process in [p for p in running if p.poll() is not None]:
            n, rho, s, log, started = running.pop(process)
            log.close()
            if process.returncode == 0:
                done += 1
                print(f"{time.ctime()}: n={n} rho={rho:g} seed={s} done in {time.time() - started:.0f} s ({done} done)",
                      flush=True)
            else:
                failed.append((n, rho, s))
                print(f"{time.ctime()}: n={n} rho={rho:g} seed={s} FAILED, see {logs}", flush=True)
    print(f"{done} jobs done, {len(failed)} failed, {len(todo)} not started", flush=True)
    for n in SIZES:
        for rho in RHOS:
            print(f"n={n} rho={rho:g}: {sum(path(n, rho, s).exists() for s in range(SEEDS))} seeds", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("job")
    one.add_argument("n", type=int)
    one.add_argument("rho", type=float)
    one.add_argument("seed", type=int)
    one.add_argument("--threads", type=int, default=1)
    many = commands.add_parser("run")
    many.add_argument("--hours", type=float, default=7)
    many.add_argument("--workers", type=int, default=os.cpu_count())
    many.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.command == "job":
        job(args.n, args.rho, args.seed, args.threads)
    else:
        run(args.hours, args.workers, args.threads)


if __name__ == "__main__":
    main()
