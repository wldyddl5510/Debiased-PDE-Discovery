"""Write results.md and figures/ from the saved results of the settings nls_random_n<n>_<method>[_wsindy_pde], and of
nls_random_n1000_<method>[_wsindy_pde]_uhat."""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments import EXPERIMENTS, NLS_LAMBDAS, NLS_LW_RADIUS, NLS_UHAT_RADIUS, NLS_WSINDY_PDE, NLS_WSINDY_PDE_RADIUS

SIZES = tuple(NLS_LW_RADIUS)
# The n of the u-hat settings run so far
PILOT_SIZE = 1000
METHODS = {"mstls": ("WSINDy", "#2a78d6", "o"), "lw": ("LW", "#eb6834", "s"), "coco": ("CoCoLasso", "#1baf7a", "^")}
FAMILIES = {"_wsindy_pde": "WSINDy paper", "": "Proposed"}
INK, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7", "#fcfcfb"
FIGURES = Path("figures")


def metrics(name):
    # L2 error and F1 score per seed, over the coefficients of both equations
    saved = np.load(Path("results")/f"{name}.npz")
    beta = np.concatenate([saved["beta_0"], saved["beta_1"]], axis=1)
    star = np.concatenate([saved["beta_star_0"], saved["beta_star_1"]])
    true_positives = np.sum((beta != 0) & (star != 0), axis=1)
    return np.linalg.norm(beta - star, axis=1), 2*true_positives/(np.sum(beta != 0, axis=1) + np.sum(star != 0))


def log_axis(values):
    # A log axis around every mean and its 2.5% and 97.5% quantiles, with its ticks
    bars = np.array([np.quantile(v, (.025, .975), axis=1) for family in values.values() for v in family.values()])
    means = np.array([v.mean(axis=1) for family in values.values() for v in family.values()])
    low, high = min(bars[bars > 0].min(), means.min())*.85, bars.max()*1.2
    ticks = [t for t in (.1, .2, .5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000) if low <= t <= high]
    return ticks, (low, high)


def plot(values, ylabel, path, yticks, ylim, log=False, reference=None):
    # Mean with its 2.5% and 97.5% quantiles per method against n, on a log n axis, one panel per family of test functions
    plt.rcParams.update({"font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"], "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(9.6, 3.9), dpi=200, facecolor=SURFACE, sharey=True)
    for ax, (family, title) in zip(axes, FAMILIES.items()):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        if log:
            ax.set_yscale("log")
        # The x positions are moved slightly apart so that the error bars stay readable
        for (method, (label, color, marker)), shift in zip(METHODS.items(), (0.97, 1, 1.03)):
            v = values[family][method]
            mean, (lower, upper) = v.mean(axis=1), np.quantile(v, (.025, .975), axis=1)
            ax.errorbar(np.array(SIZES)*shift, mean, (mean - lower, upper - mean), color=color, marker=marker,
                        ms=7, lw=1.5, elinewidth=1, capsize=2.5, mec=SURFACE, mew=1, label=label, zorder=3)
        if reference is not None:
            ax.hlines(reference, SIZES[0]*.85, SIZES[-1]*1.05, color=MUTED, lw=.8, zorder=2)
            ax.text(2200, reference*1.04, rf"$\hat\beta = 0$: {reference:.2f}", color=MUTED, fontsize=8, va="bottom", ha="center")
        ax.set(xlim=(SIZES[0]*.85, SIZES[-1]*2.1), ylim=ylim, xticks=SIZES, yticks=yticks)
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.set_yticklabels([f"{y:g}" for y in yticks])
        ax.minorticks_off()
        ax.set_title(f"{title} test functions", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("$n$", color=MUTED)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)
        ax.grid(color=GRID, lw=.6)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
    axes[0].set_ylabel(ylabel, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=3, frameon=False, labelcolor=INK, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .94))
    # Direct labels at the last n, pushed apart when two ends nearly coincide
    for ax, family in zip(axes, FAMILIES):
        last = {m: values[family][m][-1].mean() for m in METHODS}
        previous = -np.inf
        for height, method in sorted((ax.transData.transform((SIZES[-1], y))[1]*72/fig.dpi, m) for m, y in last.items()):
            previous = max(height, previous + 11)
            ax.annotate(METHODS[method][0], (SIZES[-1]*1.03, last[method]), xytext=(8, previous - height),
                        textcoords="offset points", color=INK, fontsize=9, va="center")
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES/path, facecolor=SURFACE)
    plt.close(fig)


def summary(v):
    return f"{v.mean():.2f} ({np.quantile(v, .025):.2f}, {np.quantile(v, .975):.2f})"


def table(values):
    head = [METHODS[m][0] for m in METHODS]
    rows = [f"| $n$ | Test functions | {' | '.join(head)} |", f"|---|---|{'---|'*len(head)}"]
    for i, n in enumerate(SIZES):
        for family, title in FAMILIES.items():
            rows.append(f"| {n} | {title} | {' | '.join(summary(values[family][m][i]) for m in METHODS)} |")
    return "\n".join(rows)


def pilot_table(index):
    # L2 error (index 0) or F1 score (index 1) of each u-hat setting on the seeds it has, and of the full data on the same seeds
    rows = [r"| Test functions | Method | Seeds | $U$, all $n$ points | $\hat u$, regression half |", "|---|---|---|---|---|"]
    for family, title in FAMILIES.items():
        for m in METHODS:
            name = f"nls_random_n{PILOT_SIZE}_{m}{family}"
            saved = np.load(Path("results")/f"{name}_uhat.npz")
            seeds = saved["seeds"] if "seeds" in saved else np.arange(len(saved["beta_0"]))
            rows.append(f"| {title} | {METHODS[m][0]} | {len(seeds)} | {summary(metrics(name)[index][seeds])} | {summary(metrics(name + '_uhat')[index])} |")
    return "\n".join(rows)


def proposed_table(settings):
    # The proposed test functions of every seed, as main.run builds them
    rows = ["| $n$ | $R_x$ | $R_t$ | $K$ |", "|---|---|---|---|"]
    for n, s in settings.items():
        phis = [s["test_functions"](getattr(s["pde"], s["sampling"])(n, s["rho"], seed)) for seed in range(s["iterations"])]
        radii = np.array([phi.radii for phi in phis])
        K = [len(phi.centers[0])*len(phi.centers[1]) for phi in phis]
        cells = [f"{np.median(radii[:, a]):.2f} ({radii[:, a].min():.2f}–{radii[:, a].max():.2f})" for a in (0, 1)]
        rows.append(f"| {n} | {cells[0]} | {cells[1]} | {np.median(K):.0f} ({min(K)}–{max(K)}) |")
    return "\n".join(rows)


def main():
    settings = {n: EXPERIMENTS[f"nls_random_n{n}_mstls"] for n in SIZES}
    results = {family: {m: [metrics(f"nls_random_n{n}_{m}{family}") for n in SIZES] for m in METHODS} for family in FAMILIES}
    l2 = {family: {m: np.array([r[0] for r in results[family][m]]) for m in METHODS} for family in FAMILIES}
    f1 = {family: {m: np.array([r[1] for r in results[family][m]]) for m in METHODS} for family in FAMILIES}
    setting = settings[SIZES[0]]
    seeds, rho = setting["iterations"], setting["rho"]
    zero = float(np.sqrt(sum(np.sum(b**2) for b in setting["pde"].beta_star)))
    P = len(setting["dictionary"][0].rhs)
    px, pt = setting["test_functions"](setting["pde"].random(SIZES[0], rho, 0)).degrees
    wx, wt = NLS_WSINDY_PDE.centers
    (Rx, Rt), (qx, qt) = NLS_WSINDY_PDE.radii, NLS_WSINDY_PDE.degrees
    oracle = {family: ", ".join(f"{r[n]:g} ($n={n}$)" for n in SIZES) for family, r in (("wsindy", NLS_WSINDY_PDE_RADIUS), ("proposed", NLS_LW_RADIUS))}
    n_reg, sigma = PILOT_SIZE//2, setting["pde"].rms*rho
    pilot = EXPERIMENTS[f"nls_random_n{PILOT_SIZE}_mstls_uhat"]

    plot(l2, r"$L^2$ error  $\|\hat\beta - \beta_*\|_2$", "l2_error.png", *log_axis(l2), log=True, reference=zero)
    plot(f1, "F1 score", "f1_score.png", [0, .2, .4, .6, .8, 1], (0, 1.02))

    Path("results.md").write_text(rf"""# NLS coefficient estimation: WSINDy, LW, CoCoLasso

{seeds} seeds per setting in Sections 4 and 5; seed $i$ draws the data of replication $i$, the same for every method. To reproduce, run `python main.py nls_random_n<n>_<method>[_wsindy_pde]` for each setting, then `python report.py`.

## 1. True PDE

The nonlinear Schrödinger equation with $a = {setting['pde'].a:g}$, $b = {setting['pde'].b:g}$ on $(x, t) \in [-5, 5] \times [0, \pi]$:

$$i\psi_t + a\psi_{{xx}} + b|\psi|^2\psi = 0, \qquad \psi(x, 0) = 2\thinspace\mathrm{{sech}}(x).$$

Writing $\psi = u - iv$ gives two equations with three true terms each:

$$u_t = 0.5\thinspace v_{{xx}} + u^2 v + v^3, \qquad v_t = -0.5\thinspace u_{{xx}} - u^3 - u v^2.$$

Observations are $U_i = u_*(z_i) + \varepsilon_i$, with $z_i$ uniform on the domain and noise standard deviation $\sigma_c = \rho\thinspace\mathrm{{RMS}}(u_{{*c}})$ per component, $\rho = {rho:g}$.

## 2. Test functions

Both families below have the form

$$\phi_k(x, t) = \Big(1 - \big(\tfrac{{x - x_k}}{{R_x}}\big)^2\Big)_+^{{p_x}} \Big(1 - \big(\tfrac{{t - t_k}}{{R_t}}\big)^2\Big)_+^{{p_t}}.$$

Integrals are Monte Carlo sums with $\Delta z = |\Omega|/n$, and all derivatives are moved onto the test function. The data are random in both cases.

### 2.1 WSINDy paper

The Schrödinger example of Messenger & Bortz (2021), Tables 3 and 4, unchanged at every $n$. Its values are given on a grid of $256 \times 251$: half support $m = (19, 25)$ and query-point stride $s = (5, 5)$ in grid points.

| Item | Value |
|---|---|
| Number | $K = {len(wx)} \times {len(wt)} = {len(wx)*len(wt)}$ (tensor product) |
| Radii | $R_x = {Rx:.3f}$, $R_t = {Rt:.3f}$ |
| Degrees | $p_x = {qx}$, $p_t = {qt}$ |
| Centers | $x_k = {wx[0]:.3f}, \dots, {wx[-1]:.3f}$ (spacing {wx[1] - wx[0]:.3f}), $t_k = {wt[0]:.3f}, \dots, {wt[-1]:.3f}$ (spacing {wt[1] - wt[0]:.3f}) |

### 2.2 Proposed

Built from each data set, after Appendix A of Messenger & Bortz (2021), whose spectrum and support need a grid. With $L_x = 10$ and $L_t = \pi$:

1. **Degrees.** $p_a$ is the highest derivative order in coordinate $a$ plus one, so $(p_x, p_t) = ({px}, {pt})$.
2. **Critical modes.** With $N = \sqrt{{n}}$ modes per coordinate and each component $c$ of $U$, a mode $(k, l)$ is signal if

   $$\frac{{\big|\sum_i U_{{ci}}\thinspace e^{{-2\pi i (k x_i/L_x + l t_i/L_t)}}\big|^2}}{{\sum_i U_{{ci}}^2}} \gt \log\frac{{N^2}}{{0.1}}.$$

   The left side has mean one plus the signal, whether the noise floor comes from the $z_i$ or from $\varepsilon_i$. $k_a^*$ is the last signal mode in coordinate $a$, counted up from 1 and stopping at two empty modes in a row.
3. **Radii.** $R_a = \sigma_a\sqrt{{2p_a + 3}}$ with $\sigma_a = L_a/(2\pi k_a^*)$, averaged over $u$ and $v$, and capped at $L_a/2 - L_a/N$.
4. **Centers.** A tensor lattice of spacing $L_a/50$ in $[\mathrm{{lo}}_a + R_a, \mathrm{{hi}}_a - R_a]$, which does not depend on the data.

Median (range) over the seeds:

{proposed_table(settings)}

## 3. Dictionary and estimators

$S$ is the highest order of differentiation and $J$ the highest polynomial degree, with $S = J = 6$ as in the Schrödinger example of the paper. The candidate terms of each equation are the constant, and $\partial_x^s(u^p v^q)$ with $s = 0, \dots, S$, $p + q = j$, $j = 1, \dots, J$. That gives $1 + 7 \times 27 = {P}$ terms per equation and {2*P} coefficients in total, of which 6 are nonzero.

| Method | Estimator | Choice of $\lambda$ | Use of $\beta_*$ |
|---|---|---|---|
| WSINDy | MSTLS (Messenger & Bortz 2021), without rescaling | MSTLS loss, 50 values from $10^{{-4}}$ to 1 | None |
| LW | Loh & Wainwright (2012), eq. (2.7), standardized columns | Corrected 5-fold CV (5.4), {len(NLS_LAMBDAS)} values from {NLS_LAMBDAS[0]:g} to {NLS_LAMBDAS[-1]:g} | $R$ only: {oracle['wsindy']} for the paper's test functions; {oracle['proposed']} for the proposed ones |
| CoCoLasso | Datta & Zou (2017), eq. (2.1), standardized columns | Same CV and grid as LW | None |

LW's $R$ is an oracle value: the median of $\sqrt{{3}}\thinspace\Vert D\beta_*\Vert_2$ over 20 seeds and both equations. The CV score (5.4) comes from the CoCoLasso paper, and LW uses the same score. LW and CoCoLasso use $\widehat\Sigma_W$, estimated by a sample covariance, so they do not need $\sigma$.

## 4. $L^2$ error

$\Vert\hat\beta - \beta_*\Vert_2$ over all {2*P} coefficients of both equations. The reference line is the value at $\hat\beta = 0$, {zero:.2f}. The vertical axis is logarithmic, points show the mean and bars span the 2.5% to 97.5% quantiles across seeds, and points are shifted slightly horizontally to avoid overlap.

![L2 error](figures/l2_error.png)

Mean (2.5% quantile, 97.5% quantile) over {seeds} seeds:

{table(l2)}

## 5. F1 score

With $\hat S = \lbrace\ell : \hat\beta_\ell \neq 0\rbrace$ and $S_\ast = \lbrace\ell : \beta_{{\ast\ell}} \neq 0\rbrace$, $F_1 = 2|\hat S \cap S_\ast|/(|\hat S| + |S_\ast|)$ over all {2*P} coefficients of both equations. Points show the mean and bars span the 2.5% to 97.5% quantiles across seeds, and points are shifted slightly horizontally to avoid overlap.

![F1 score](figures/f1_score.png)

Mean (2.5% quantile, 97.5% quantile) over {seeds} seeds:

{table(f1)}

## 6. $U$ replaced by a pilot $\hat u$, $n = {PILOT_SIZE}$

Each seed's {PILOT_SIZE} points are split in half. The pilot half is taken to have produced $\hat u$, and the estimators see only the other half, the regression half of $n_{{\rm reg}} = {n_reg}$ points, with $\hat u(z_i)$ in place of $U_i$ at its own $z_i$:

$$\hat u_c = u_{{*c}} + \zeta\thinspace n_{{\rm reg}}^{{-1/3}} g_c, \qquad g_c(x, t) = \sum_{{k, l = 0}}^{{8}} a_{{ckl}} \cos\frac{{\pi k (x + 5)}}{{L_x}} \cos\frac{{\pi l t}}{{L_t}},$$

with $\zeta = 1$ and $a_{{ckl}} \sim N(0, 1)$, drawn afresh for each seed and scaled so that $g_c$ has RMS 1 over the domain. The pilot error therefore has RMS ${n_reg}^{{-1/3}} = {n_reg**(-1/3):.3f}$ in each component, against the noise standard deviation $\sigma = ({sigma[0]:.3f}, {sigma[1]:.3f})$ of $U$. Unlike the noise, it is smooth, so the test functions do not average it out.

$X_n$, $Y_n$ and $\widehat\Sigma_W$ are computed from $\hat u$, so $\widehat\Sigma_W$ covers only the Monte Carlo variation over the $z_i$. The paper's test functions are unchanged, and the proposed ones are built from the regression half's $U$. LW's $R$ follows the oracle rule of Section 3 on the regression half with $\hat u$: {NLS_UHAT_RADIUS['_wsindy_pde'][PILOT_SIZE]:g} for the paper's test functions and {NLS_UHAT_RADIUS[''][PILOT_SIZE]:g} for the proposed ones.

The run was stopped after a fixed time, so each setting has only the seeds that finished. The column $U$ repeats Sections 4 and 5 on the same seeds. `python main.py nls_random_n{PILOT_SIZE}_<method>[_wsindy_pde]_uhat` runs all {pilot['iterations']} seeds of a setting.

### 6.1 $L^2$ error

Mean (2.5% quantile, 97.5% quantile) over the seeds:

{pilot_table(0)}

### 6.2 F1 score

Mean (2.5% quantile, 97.5% quantile) over the seeds:

{pilot_table(1)}
""", encoding="utf-8")
    print("wrote results.md")


if __name__ == "__main__":
    main()
