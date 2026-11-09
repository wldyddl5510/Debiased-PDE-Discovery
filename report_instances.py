"""Write results_instances.md and figures/instances_*.png from the seeds that instances.py saved.
python report_instances.py"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import NLS_LAMBDAS, NLS_WSINDY_PDE, NLS_WSINDY_PDE_RADII
from instances import GRAMS, INPUTS, INSTANCES, OUT, RHOS, SEEDS, SIZES, path

# Color follows the estimator, slots 1-5 of the reference palette in order (blue, orange and aqua as in report.py);
# line style and marker follow the input
COLORS = {"mstls": "#2a78d6", "lw": "#eb6834", "coco": "#1baf7a", "lasso": "#eda100", "bss": "#e87ba4"}
STYLES = {"U": ("-", "o"), "plugin": ("-", "o"), "db": ("--", "D")}
ROWS = {"$U_i$": ["wsindy", "lasso", "lw", "coco", "bss"],
        r"$\hat u$": ["plugin_wsindy", "db_wsindy", "plugin_bss", "db_bss"]}
GRAM_LABELS = {"truth": r"$X(u_*)$ on the grid, reference", "U": "$X_n(U)$", "U_eiv": r"$X_n(U)$, $X'X/K - \widehat\Sigma_W$",
               "plugin": r"$X_n(\hat u)$, u-hat (all n)", "db": r"$X^{\mathrm{db}}_n(\hat u)$, u-hat (split)"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FIGURES = Path("figures")


def load(n, rho):
    # {key: array over the saved seeds of (n, rho), in seed order}, and the seeds
    seeds = [s for s in range(SEEDS) if path(n, rho, s).exists()]
    files = [np.load(path(n, rho, s)) for s in seeds]
    return {key: np.array([f[key] for f in files]) for key in (files[0].files if files else [])}, seeds


def metrics(saved, key):
    # L2 error and F1 score per seed, over the coefficients of both equations
    beta = np.concatenate([saved[f"beta_{key}_0"], saved[f"beta_{key}_1"]], axis=1)
    star = np.concatenate([saved["beta_star_0"][0], saved["beta_star_1"][0]])
    true_positives = np.sum((beta != 0) & (star != 0), axis=1)
    return np.linalg.norm(beta - star, axis=1), 2*true_positives/(np.sum(beta != 0, axis=1) + np.sum(star != 0))


def summary(v):
    return f"{v.mean():.2f} ({np.quantile(v, .025):.2f}, {np.quantile(v, .975):.2f})" if len(v) else "–"


def plot(values, ylabel, name, log=False, reference=None, ylim=None):
    # Mean with its 2.5% and 97.5% quantiles against n, a column per rho and a row per input of the instances
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(len(ROWS), len(RHOS), figsize=(11, 7), dpi=200, facecolor=SURFACE, sharex=True, sharey=True)
    for i, (row, keys) in enumerate(ROWS.items()):
        for j, rho in enumerate(RHOS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.set_xscale("log")
            if log:
                ax.set_yscale("log")
            for key, shift in zip(keys, np.linspace(0.94, 1.06, len(keys))):
                label, source, method = INSTANCES[key]
                v = [values[n, rho][key] for n in SIZES]
                ok = [len(x) > 0 for x in v]
                sizes = np.array(SIZES)[ok]*shift
                mean = np.array([x.mean() for x, o in zip(v, ok) if o])
                bars = np.array([np.quantile(x, (.025, .975)) for x, o in zip(v, ok) if o]).reshape(-1, 2).T
                style, marker = STYLES[source]
                ax.errorbar(sizes, mean, (mean - bars[0], bars[1] - mean), color=COLORS[method], ls=style, marker=marker,
                            ms=6, lw=1.5, elinewidth=1, capsize=2.5, mec=SURFACE, mew=1, label=label, zorder=3)
            if reference is not None:
                ax.axhline(reference, color=MUTED, lw=.8, zorder=2, label=rf"$\hat\beta = 0$: {reference:.2f}")
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xticks(SIZES)
            ax.set_xticklabels([str(n) for n in SIZES])
            ax.minorticks_off()
            ax.grid(axis="y", color=GRID, lw=.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color(AXIS)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_title(rf"{row}, $\rho = {rho:g}$", color=INK, fontsize=10, loc="left")
            if i == len(ROWS) - 1:
                ax.set_xlabel("$n$", color=MUTED)
            if j == 0:
                ax.set_ylabel(ylabel, color=MUTED)
        axes[i, -1].legend(frameon=False, fontsize=8, labelcolor=INK, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES/name, facecolor=SURFACE)
    plt.close(fig)


def instance_table(values, rho):
    rows = [f"| Instance | {' | '.join(f'$n = {n}$' for n in SIZES)} |", f"|---|{'---|'*len(SIZES)}"]
    rows += [f"| {label} | {' | '.join(summary(values[n, rho][key]) for n in SIZES)} |" for key, (label, _, _) in INSTANCES.items()]
    return "\n".join(rows)


def irrepresentability_table(saved, rho):
    # Median eta / median mu over the seeds and both equations, and the share of eta < 1
    rows = [f"| Gram matrix | {' | '.join(f'$n = {n}$' for n in SIZES)} |", f"|---|{'---|'*len(SIZES)}"]
    for label in GRAMS:
        cells = []
        for n in SIZES:
            s = saved[n, rho][0]
            if not s:
                cells.append("–")
                continue
            v = np.concatenate([s[f"irr_{label}_0"], s[f"irr_{label}_1"]])
            cells.append(f"{np.median(v[:, 0]):.2f} / {np.median(v[:, 1]):.2f} ({np.mean(v[:, 0] < 1):.0%})")
        rows.append(f"| {GRAM_LABELS[label]} | {' | '.join(cells)} |")
    return "\n".join(rows)


def setting_table(saved, head, cells):
    # One row per (n, rho) with its number of seeds, and the cells(s) of its saved arrays s
    rows = [f"| $n$ | $\\rho$ | seeds | {' | '.join(head)} |", f"|---|---|---|{'---|'*len(head)}"]
    for n in SIZES:
        for rho in RHOS:
            s, seeds = saved[n, rho]
            rows.append(f"| {n} | {rho:g} | {len(seeds)} | {' | '.join(cells(s) if seeds else ['–']*len(head))} |")
    return "\n".join(rows)


def pilot_table(saved):
    # Median over the seeds of the epochs and of the relative L^2 error of (u-hat, v-hat), each component
    head = ["epochs, all $n$", "epochs, split", "rel. $L^2$ error, all $n$", "rel. $L^2$ error, split"]
    pair = lambda v: f"{np.median(v[:, 0]):.3f}, {np.median(v[:, 1]):.3f}"
    return setting_table(saved, head, lambda s: [f"{np.median(s['epochs_all']):.0f}", f"{np.median(s['epochs_split']):.0f}",
                                                  pair(s["l2_all"]), pair(s["l2_split"])])


def distance_table(saved):
    # Median over the seeds and both equations of ||A - A(u_*)||/||A(u_*)|| for X on the true terms and for Y
    head = [f"{a}, {INPUTS[k]}" for a in ("$X$ true terms", "$Y$") for k in INPUTS]
    dist = lambda s, k, a: np.median(np.concatenate([s[f"dist_{k}_0"][:, a], s[f"dist_{k}_1"][:, a]]))
    return setting_table(saved, head, lambda s: [f"{dist(s, k, a):.2f}" for a in (1, 2) for k in INPUTS])


def bss_table(saved):
    # Share of the best-subset paths whose MIO is certified optimal (gap <= 1e-4) at each k, per input, over all settings
    rows = [f"| Input | {' | '.join(f'$k = {k}$' for k in range(1, 11))} |", f"|---|{'---|'*10}"]
    for source, label in INPUTS.items():
        gaps = [s[f"bss_gap_{source}_{q}"] for s, seeds in saved.values() if seeds for q in (0, 1)]
        if gaps:
            gaps = np.concatenate(gaps)
            rows.append(f"| {label} | {' | '.join(f'{np.mean(g <= 1e-4):.0%}' for g in gaps.T)} |")
    return "\n".join(rows)


def time_table(saved):
    # Median minutes per seed, for both equations: the two PINN fits and each instance, over the settings of each n
    names = {"all": "PINN, u-hat (all n)", "split": "PINN, u-hat (split)", **{k: v[0] for k, v in INSTANCES.items()}}
    rows = [f"| Step | {' | '.join(f'$n = {n}$' for n in SIZES)} |", f"|---|{'---|'*len(SIZES)}"]
    for key, name in names.items():
        t = [np.concatenate([saved[n, rho][0][f"time_{key}"] for rho in RHOS if saved[n, rho][1]] or [[np.nan]]) for n in SIZES]
        rows.append(f"| {name} | {' | '.join(f'{np.median(v)/60:.1f}' for v in t)} |")
    return "\n".join(rows)


def main():
    saved = {(n, rho): load(n, rho) for n in SIZES for rho in RHOS}
    l2, f1 = {}, {}
    for (n, rho), (s, seeds) in saved.items():
        l2[n, rho], f1[n, rho] = {}, {}
        for key in INSTANCES:
            l2[n, rho][key], f1[n, rho][key] = metrics(s, key) if seeds else (np.array([]), np.array([]))
    counts = {k: len(seeds) for k, (_, seeds) in saved.items()}
    zero = next(float(np.sqrt(np.sum(s["beta_star_0"][0]**2) + np.sum(s["beta_star_1"][0]**2))) for s, seeds in saved.values() if seeds)

    plot(l2, r"$\|\hat\beta - \beta_*\|_2$", "instances_l2_error.png", log=True, reference=zero)
    plot(f1, "F1 score", "instances_f1_score.png", ylim=(-0.03, 1.03))

    # The pieces of the page, computed here as Python 3.11 allows no backslash inside an f-string expression
    count_rows = "\n".join(f"| {n} | {' | '.join(str(counts[n, rho]) for rho in RHOS)} |" for n in SIZES)
    radius_rows = "\n".join(f"  | {n} | {' | '.join(f'{NLS_WSINDY_PDE_RADII[n, rho]:g}' for rho in RHOS)} |" for n in SIZES)
    instance_rows = "\n".join(f"| {label} | {INPUTS[source]} | {method} |" for label, source, method in INSTANCES.values())
    rho_head = " | ".join(rf"$\rho = {rho:g}$" for rho in RHOS)
    rule = "---|"*len(RHOS)
    sizes, rhos = ", ".join(map(str, SIZES)), ", ".join(f"{r:g}" for r in RHOS)
    by_rho = lambda table: "\n\n".join(rf"### $\rho = {rho:g}$" + "\n\n" + table(rho) for rho in RHOS)
    l2_tables, f1_tables = by_rho(lambda rho: instance_table(l2, rho)), by_rho(lambda rho: instance_table(f1, rho))
    irr_tables = by_rho(lambda rho: irrepresentability_table(saved, rho))
    wx, wt = NLS_WSINDY_PDE.centers
    (Rx, Rt), (qx, qt) = NLS_WSINDY_PDE.radii, NLS_WSINDY_PDE.degrees

    Path("results_instances.md").write_text(rf"""# NLS: the 9 instances of single_instance.ipynb over $n$ and $\rho$

Every instance of `single_instance.ipynb`, for $n = {sizes}$ and $\rho = {rhos}$. Seed $i$ draws the data of replication $i$, the same for every instance of a setting. The run was stopped after 7 hours, so each setting has only the seeds that finished all 9 instances:

| $n$ | {rho_head} |
|---|{rule}
{count_rows}

With this few seeds, the 2.5% and 97.5% quantiles below are close to the smallest and largest values. To reproduce, run `python instances.py run --hours 7`, which continues from the saved seeds, then `python report_instances.py`.

## 1. Setup

As in the notebook, with $n$ and $\rho$ varied:

* **PDE and data.** The nonlinear Schrödinger equation with $a = 0.5$, $b = 1$ on $[-5, 5] \times [0, \pi]$, $\psi(x, 0) = 2\thinspace\mathrm{{sech}}(x)$, three true terms in each of the two equations. $z_i$ uniform on the domain, $U_i = u_*(z_i) + \varepsilon_i$ with $\sigma_c = \rho\thinspace\mathrm{{RMS}}(u_{{*c}})$.
* **Test functions.** The Schrödinger example of Messenger & Bortz (2021): $K = {len(wx)} \times {len(wt)} = {len(wx)*len(wt)}$, $R_x = {Rx:.3f}$, $R_t = {Rt:.3f}$, $p = ({qx}, {qt})$.
* **Dictionary.** The constant and $\partial_x^s(u^p v^q)$, $s = 0, \dots, 6$, $p + q = 1, \dots, 6$: 190 terms per equation.
* **u-hat.** A tanh network of width $(64, 64, 64)$, Adam with learning rate $10^{{-3}}$ on the full batch, $\kappa = 0$, at most 5000 epochs chosen on 20% held out, then refit on all its observations. u-hat (all n) is fit to all $n$ observations and summed on the $256 \times 251$ grid; u-hat (split) is fit to one half (`data.split()`), and the other half gives the first-order correction of $X^{{\mathrm{{db}}}}_n$, $Y^{{\mathrm{{db}}}}_n$.

| Instance | Input | Estimator |
|---|---|---|
{instance_rows}

* **mstls**: MSTLS of Messenger & Bortz (2021), without their rescaling, $\lambda$ by the MSTLS loss over 50 values in $[10^{{-4}}, 1]$.
* **lasso**: $\frac{{1}}{{2K}}\Vert Y - X\beta\Vert_2^2 + \lambda\Vert D\beta\Vert_1$, $\lambda$ by the MSTLS loss over 50 values on its path.
* **lw**: Loh & Wainwright (2012), eq. (2.7), standardized columns, $\lambda$ by the corrected 5-fold CV of Datta & Zou (2017) over {len(NLS_LAMBDAS)} values in $[{NLS_LAMBDAS[0]:g}, {NLS_LAMBDAS[-1]:g}]$. $R$ is an oracle value, the median of $\sqrt 3\thinspace\Vert D\beta_*\Vert_2$ over 20 seeds and both equations, computed for each $(n, \rho)$:

  | $n$ | {rho_head} |
  |---|{rule}
{radius_rows}

* **coco**: CoCoLasso (Datta & Zou 2017), the same CV, no radius.
* **bss**: best subset selection of Bertsimas, King & Mazumder (2016) for $k = 1, \dots, 10$ by Gurobi, 30 s per MIO, $k$ by the MSTLS loss.

## 2. $L^2$ error

$\Vert\hat\beta - \beta_*\Vert_2$ over the 380 coefficients of both equations; the gray line is its value at $\hat\beta = 0$, {zero:.2f}. Points show the mean and bars span the 2.5% to 97.5% quantiles across seeds; the vertical axis is logarithmic. Color shows the estimator; in the bottom row, solid lines are u-hat (all n) and dashed lines u-hat (split), debiased.

![L2 error](figures/instances_l2_error.png)

Mean (2.5% quantile, 97.5% quantile) over the seeds:

{l2_tables}

## 3. F1 score

$F_1 = 2|\hat S \cap S_*|/(|\hat S| + |S_*|)$ over the 380 coefficients of both equations, with $\hat S$ and $S_*$ the nonzero entries of $\hat\beta$ and $\beta_*$.

![F1 score](figures/instances_f1_score.png)

Mean (2.5% quantile, 97.5% quantile) over the seeds:

{f1_tables}

## 4. Irrepresentable condition

For one equation with true terms $S$ and Gram matrix $C$ of the standardized columns, $\eta = \Vert C_{{S^cS}} C_{{SS}}^{{-1}}\mathrm{{sign}}(\beta_{{*,S}})\Vert_\infty$ (Zhao & Yu 2006) and its sign-free version $\mu = \Vert C_{{S^cS}} C_{{SS}}^{{-1}}\Vert_\infty$ (Wainwright 2009). The lasso can select $S$ with the correct signs only if $\eta < 1$. Each cell is the median $\eta$ / median $\mu$ over the seeds and both equations, and in parentheses the share of $\eta < 1$.

{irr_tables}

## 5. u-hat and the weak forms

Median over the seeds of the number of epochs and of $\Vert\hat u - u_*\Vert_{{L^2(\Omega)}}/\Vert u_*\Vert_{{L^2(\Omega)}}$, for $u$ and $v$:

{pilot_table(saved)}

Median over the seeds and both equations of $\Vert A - A(u_*)\Vert/\Vert A(u_*)\Vert$, for $X$ on the true terms and for $Y$, with $A(u_*)$ from $u_*$ summed on the $256 \times 251$ grid:

{distance_table(saved)}

## 6. Best subset: certified optimal MIOs

Share of the paths, over all settings, seeds and both equations, whose MIO at $k$ ended with a gap of at most $10^{{-4}}$ within 30 s.

{bss_table(saved)}

## 7. Run time

Median minutes per seed with one thread, both equations together, over the three $\rho$ of each $n$; 10 seeds ran at a time on a 10-core Apple M4.

{time_table(saved)}
""")
    print("wrote results_instances.md")


if __name__ == "__main__":
    main()
