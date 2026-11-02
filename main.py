"""Run one setting of experiments.py: python main.py <name>"""
import argparse
from pathlib import Path

import numpy as np

from experiments import EXPERIMENTS
from weak_form import WeakForm


def run(pde, sampling, size, rho, iterations, estimator, dictionary, test_functions, pilot=None, m=None, use_original_data=False):
    # beta-hat per equation, each of shape (iterations, P)
    betas = [[] for _ in dictionary]
    for seed in range(iterations):
        data = getattr(pde, sampling)(size, rho, seed)
        if pilot is None:
            wf = WeakForm(data, test_functions)
        else:
            # Split first, so that u-hat comes from one half and beta-hat from the other; u-hat then replaces U
            pilot_data, data = data.split()
            wf = WeakForm(data, test_functions, pilot(pilot_data, seed=(seed, 1)), m, use_original_data, seed=(seed, 2))
        for q, equation in enumerate(dictionary):
            betas[q].append(estimator.estimate(wf, equation))
    return [np.array(b) for b in betas]


def truth(pde, q, equation):
    # beta_* on the columns of the dictionary, 0 for the terms not in the PDE
    known = dict(zip(pde.equations[q].rhs, pde.beta_star[q]))
    return np.array([known.get(term, 0.) for term in equation.rhs])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=EXPERIMENTS)
    name = parser.parse_args().experiment
    setting = EXPERIMENTS[name]

    betas = run(**setting)
    truths = [truth(setting["pde"], q, eq) for q, eq in enumerate(setting["dictionary"])]
    for q, (equation, beta, star) in enumerate(zip(setting["dictionary"], betas, truths)):
        print(f"equation {q}: lhs {equation.lhs}")
        print(f"{'powers':>10} {'derivative':>10} {'beta*':>8} {'mean':>9} {'sd':>8} {'selected':>8}")
        for l, term in enumerate(equation.rhs):
            print(f"{str(term.powers):>10} {str(term.derivative):>10} {star[l]:8.3f} {beta[:, l].mean():9.4f} "
                  f"{beta[:, l].std(ddof=1):8.4f} {np.mean(beta[:, l] != 0):8.2f}")

    path = Path("results")/f"{name}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, **{f"beta_{q}": b for q, b in enumerate(betas)}, **{f"beta_star_{q}": s for q, s in enumerate(truths)})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
