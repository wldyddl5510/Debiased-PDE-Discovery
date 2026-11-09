"""Run one setting of experiments.py: python main.py <name>"""
import argparse
from pathlib import Path

import numpy as np

from experiments import EXPERIMENTS
from weak_form import CrossFitWeakForm, WeakForm


def run(pde, sampling, size, rho, iterations, estimator, dictionary, test_functions, pilot=None, points="random", m=None,
        folds=None, debias=True, cross_fit=True):
    # beta-hat per estimator and equation, each of shape (iterations, P); estimator is one, or a dict of them by name
    estimators = estimator if isinstance(estimator, dict) else {None: estimator}
    betas = {name: [[] for _ in dictionary] for name in estimators}
    for seed in range(iterations):
        data = getattr(pde, sampling)(size, rho, seed)
        if pilot is None:
            wf = WeakForm(data, test_functions)
        elif folds is None:
            # u-hat is fit to all n observations, and X_n, Y_n are then computed from u-hat at m new points
            wf = WeakForm(data, test_functions, pilot(data, seed=(seed, 1)), points, m, seed=(seed, 2))
        else:
            # u-hat_f is fit outside fold f, and X_n, Y_n average its terms at m new points, corrected on fold f if debias;
            # with cross_fit=False only the last fold is held out (the sample splitting of Algorithm 2)
            wf = CrossFitWeakForm(data, test_functions, pilot, points, m, folds, debias, seed, cross_fit)
        for name, est in estimators.items():
            for q, equation in enumerate(dictionary):
                betas[name][q].append(est.estimate(wf, equation))
    out = {name: [np.array(b) for b in bs] for name, bs in betas.items()}
    return out if isinstance(estimator, dict) else out[None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    name = parser.parse_args().experiment
    setting = EXPERIMENTS[name]

    runs = run(**setting)
    # One file per estimator: results/<name>.npz, or results/<name>_<key>.npz for a dict of estimators
    runs = {f"{name}_{key}": b for key, b in runs.items()} if isinstance(setting["estimator"], dict) else {name: runs}
    truths = [setting["pde"].coefficients(eq) for eq in setting["dictionary"]]
    for saved, betas in runs.items():
        print(saved)
        for q, (equation, beta, star) in enumerate(zip(setting["dictionary"], betas, truths)):
            print(f"equation {q}: lhs {equation.lhs}")
            print(f"{'powers':>10} {'derivative':>10} {'beta*':>8} {'mean':>9} {'sd':>8} {'selected':>8}")
            for l, term in enumerate(equation.rhs):
                print(f"{str(term.powers):>10} {str(term.derivative):>10} {star[l]:8.3f} {beta[:, l].mean():9.4f} "
                      f"{beta[:, l].std(ddof=1):8.4f} {np.mean(beta[:, l] != 0):8.2f}")

        path = Path("results")/f"{saved}.npz"
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **{f"beta_{q}": b for q, b in enumerate(betas)}, **{f"beta_star_{q}": s for q, s in enumerate(truths)})
        print(f"saved {path}")


if __name__ == "__main__":
    main()
