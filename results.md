# NLS coefficient estimation: WSINDy, LW, CoCoLasso

100 seeds per setting in Sections 4 and 5; seed $i$ draws the data of replication $i$, the same for every method. To reproduce, run `python main.py nls_random_n<n>_<method>[_wsindy_pde]` for each setting, then `python report.py`.

## 1. True PDE

The nonlinear Schrödinger equation with $a = 0.5$, $b = 1$ on $(x, t) \in [-5, 5] \times [0, \pi]$:

$$i\psi_t + a\psi_{xx} + b|\psi|^2\psi = 0, \qquad \psi(x, 0) = 2\thinspace\mathrm{sech}(x).$$

Writing $\psi = u - iv$ gives two equations with three true terms each:

$$u_t = 0.5\thinspace v_{xx} + u^2 v + v^3, \qquad v_t = -0.5\thinspace u_{xx} - u^3 - u v^2.$$

Observations are $U_i = u_*(z_i) + \varepsilon_i$, with $z_i$ uniform on the domain and noise standard deviation $\sigma_c = \rho\thinspace\mathrm{RMS}(u_{*c})$ per component, $\rho = 0.1$.

## 2. Test functions

Both families below have the form

$$\phi_k(x, t) = \Big(1 - \big(\tfrac{x - x_k}{R_x}\big)^2\Big)_+^{p_x} \Big(1 - \big(\tfrac{t - t_k}{R_t}\big)^2\Big)_+^{p_t}.$$

Integrals are Monte Carlo sums with $\Delta z = |\Omega|/n$, and all derivatives are moved onto the test function. The data are random in both cases.

### 2.1 WSINDy paper

The Schrödinger example of Messenger & Bortz (2021), Tables 3 and 4, unchanged at every $n$. Its values are given on a grid of $256 \times 251$: half support $m = (19, 25)$ and query-point stride $s = (5, 5)$ in grid points.

| Item | Value |
|---|---|
| Number | $K = 44 \times 41 = 1804$ (tensor product) |
| Radii | $R_x = 0.742$, $R_t = 0.314$ |
| Degrees | $p_x = 11$, $p_t = 10$ |
| Centers | $x_k = -4.258, \dots, 4.141$ (spacing 0.195), $t_k = 0.314, \dots, 2.827$ (spacing 0.063) |

### 2.2 Proposed

Built from each data set, after Appendix A of Messenger & Bortz (2021), whose spectrum and support need a grid. With $L_x = 10$ and $L_t = \pi$:

1. **Degrees.** $p_a$ is the highest derivative order in coordinate $a$ plus one, so $(p_x, p_t) = (7, 2)$.
2. **Critical modes.** With $N = \sqrt{n}$ modes per coordinate and each component $c$ of $U$, a mode $(k, l)$ is signal if

   $$\frac{\big|\sum_i U_{ci}\thinspace e^{-2\pi i (k x_i/L_x + l t_i/L_t)}\big|^2}{\sum_i U_{ci}^2} \gt \log\frac{N^2}{0.1}.$$

   The left side has mean one plus the signal, whether the noise floor comes from the $z_i$ or from $\varepsilon_i$. $k_a^*$ is the last signal mode in coordinate $a$, counted up from 1 and stopping at two empty modes in a row.
3. **Radii.** $R_a = \sigma_a\sqrt{2p_a + 3}$ with $\sigma_a = L_a/(2\pi k_a^*)$, averaged over $u$ and $v$, and capped at $L_a/2 - L_a/N$.
4. **Centers.** A tensor lattice of spacing $L_a/50$ in $[\mathrm{lo}_a + R_a, \mathrm{hi}_a - R_a]$, which does not depend on the data.

Median (range) over the seeds:

| $n$ | $R_x$ | $R_t$ | $K$ |
|---|---|---|---|
| 1000 | 1.64 (1.29–2.19) | 0.39 (0.33–0.66) | 1292 (957–1520) |
| 5000 | 1.09 (0.88–1.31) | 0.33 (0.28–0.33) | 1600 (1480–1680) |
| 10000 | 0.94 (0.77–1.09) | 0.33 (0.21–0.33) | 1640 (1600–1848) |

## 3. Dictionary and estimators

$S$ is the highest order of differentiation and $J$ the highest polynomial degree, with $S = J = 6$ as in the Schrödinger example of the paper. The candidate terms of each equation are the constant, and $\partial_x^s(u^p v^q)$ with $s = 0, \dots, S$, $p + q = j$, $j = 1, \dots, J$. That gives $1 + 7 \times 27 = 190$ terms per equation and 380 coefficients in total, of which 6 are nonzero.

| Method | Estimator | Choice of $\lambda$ | Use of $\beta_*$ |
|---|---|---|---|
| WSINDy | MSTLS (Messenger & Bortz 2021), without rescaling | MSTLS loss, 50 values from $10^{-4}$ to 1 | None |
| LW | Loh & Wainwright (2012), eq. (2.7), standardized columns | Corrected 5-fold CV (5.4), 18 values from 0.005 to 1 | $R$ only: 0.79 ($n=1000$), 0.47 ($n=5000$), 0.43 ($n=10000$) for the paper's test functions; 1.63 ($n=1000$), 1.03 ($n=5000$), 0.91 ($n=10000$) for the proposed ones |
| CoCoLasso | Datta & Zou (2017), eq. (2.1), standardized columns | Same CV and grid as LW | None |

LW's $R$ is an oracle value: the median of $\sqrt{3}\thinspace\Vert D\beta_*\Vert_2$ over 20 seeds and both equations. The CV score (5.4) comes from the CoCoLasso paper, and LW uses the same score. LW and CoCoLasso use $\widehat\Sigma_W$, estimated by a sample covariance, so they do not need $\sigma$.

## 4. $L^2$ error

$\Vert\hat\beta - \beta_*\Vert_2$ over all 380 coefficients of both equations. The reference line is the value at $\hat\beta = 0$, 2.12. The vertical axis is logarithmic, points show the mean and bars span the 2.5% to 97.5% quantiles across seeds, and points are shifted slightly horizontally to avoid overlap.

![L2 error](figures/l2_error.png)

Mean (2.5% quantile, 97.5% quantile) over 100 seeds:

| $n$ | Test functions | WSINDy | LW | CoCoLasso |
|---|---|---|---|---|
| 1000 | WSINDy paper | 19.22 (7.71, 37.50) | 1.97 (1.58, 2.51) | 1.97 (1.55, 2.67) |
| 1000 | Proposed | 10.32 (3.44, 21.52) | 2.02 (1.60, 2.62) | 1.85 (1.39, 2.75) |
| 5000 | WSINDy paper | 9.81 (6.39, 15.96) | 1.78 (1.31, 2.25) | 1.74 (1.26, 2.40) |
| 5000 | Proposed | 6.49 (1.90, 12.26) | 1.62 (0.99, 2.14) | 1.44 (0.94, 2.25) |
| 10000 | WSINDy paper | 7.50 (4.06, 11.97) | 1.54 (1.13, 1.85) | 1.44 (1.04, 1.79) |
| 10000 | Proposed | 4.99 (0.75, 10.57) | 1.16 (0.63, 1.70) | 0.96 (0.54, 1.49) |

## 5. F1 score

With $\hat S = \lbrace\ell : \hat\beta_\ell \neq 0\rbrace$ and $S_\ast = \lbrace\ell : \beta_{\ast\ell} \neq 0\rbrace$, $F_1 = 2|\hat S \cap S_\ast|/(|\hat S| + |S_\ast|)$ over all 380 coefficients of both equations. Points show the mean and bars span the 2.5% to 97.5% quantiles across seeds, and points are shifted slightly horizontally to avoid overlap.

![F1 score](figures/f1_score.png)

Mean (2.5% quantile, 97.5% quantile) over 100 seeds:

| $n$ | Test functions | WSINDy | LW | CoCoLasso |
|---|---|---|---|---|
| 1000 | WSINDy paper | 0.11 (0.06, 0.17) | 0.37 (0.00, 0.67) | 0.38 (0.15, 0.63) |
| 1000 | Proposed | 0.13 (0.03, 0.31) | 0.29 (0.00, 0.50) | 0.43 (0.00, 0.70) |
| 5000 | WSINDy paper | 0.19 (0.11, 0.26) | 0.42 (0.22, 0.67) | 0.37 (0.19, 0.67) |
| 5000 | Proposed | 0.31 (0.14, 0.57) | 0.44 (0.14, 0.72) | 0.48 (0.27, 0.73) |
| 10000 | WSINDy paper | 0.24 (0.16, 0.38) | 0.42 (0.28, 0.56) | 0.40 (0.25, 0.54) |
| 10000 | Proposed | 0.39 (0.16, 0.91) | 0.57 (0.33, 0.71) | 0.52 (0.36, 0.67) |

## 6. $U$ replaced by a pilot $\hat u$, $n = 1000$

Each seed's 1000 points are split in half. The pilot half is taken to have produced $\hat u$, and the estimators see only the other half, the regression half of $n_{\rm reg} = 500$ points, with $\hat u(z_i)$ in place of $U_i$ at its own $z_i$:

$$\hat u_c = u_{*c} + \zeta\thinspace n_{\rm reg}^{-1/3} g_c, \qquad g_c(x, t) = \sum_{k, l = 0}^{8} a_{ckl} \cos\frac{\pi k (x + 5)}{L_x} \cos\frac{\pi l t}{L_t},$$

with $\zeta = 1$ and $a_{ckl} \sim N(0, 1)$, drawn afresh for each seed and scaled so that $g_c$ has RMS 1 over the domain. The pilot error therefore has RMS $500^{-1/3} = 0.126$ in each component, against the noise standard deviation $\sigma = (0.063, 0.063)$ of $U$. Unlike the noise, it is smooth, so the test functions do not average it out.

$X_n$, $Y_n$ and $\widehat\Sigma_W$ are computed from $\hat u$, so $\widehat\Sigma_W$ covers only the Monte Carlo variation over the $z_i$. The paper's test functions are unchanged, and the proposed ones are built from the regression half's $U$. LW's $R$ follows the oracle rule of Section 3 on the regression half with $\hat u$: 1.08 for the paper's test functions and 2.23 for the proposed ones.

The run was stopped after a fixed time, so each setting has only the seeds that finished. The column $U$ repeats Sections 4 and 5 on the same seeds. `python main.py nls_random_n1000_<method>[_wsindy_pde]_uhat` runs all 10 seeds of a setting.

### 6.1 $L^2$ error

Mean (2.5% quantile, 97.5% quantile) over the seeds:

| Test functions | Method | Seeds | $U$, all $n$ points | $\hat u$, regression half |
|---|---|---|---|---|
| WSINDy paper | WSINDy | 10 | 17.79 (10.87, 23.32) | 37.35 (24.59, 54.21) |
| WSINDy paper | LW | 10 | 1.88 (1.60, 2.23) | 2.04 (1.76, 2.19) |
| WSINDy paper | CoCoLasso | 7 | 1.85 (1.66, 2.20) | 1.93 (1.70, 2.30) |
| Proposed | WSINDy | 10 | 9.81 (2.93, 15.48) | 21.54 (5.72, 58.69) |
| Proposed | LW | 10 | 1.96 (1.62, 2.38) | 2.23 (1.83, 2.84) |
| Proposed | CoCoLasso | 10 | 1.62 (1.13, 1.92) | 1.99 (1.64, 2.41) |

### 6.2 F1 score

Mean (2.5% quantile, 97.5% quantile) over the seeds:

| Test functions | Method | Seeds | $U$, all $n$ points | $\hat u$, regression half |
|---|---|---|---|---|
| WSINDy paper | WSINDy | 10 | 0.13 (0.10, 0.16) | 0.07 (0.05, 0.11) |
| WSINDy paper | LW | 10 | 0.38 (0.21, 0.60) | 0.20 (0.00, 0.50) |
| WSINDy paper | CoCoLasso | 7 | 0.35 (0.22, 0.44) | 0.29 (0.00, 0.50) |
| Proposed | WSINDy | 10 | 0.18 (0.10, 0.29) | 0.05 (0.01, 0.12) |
| Proposed | LW | 10 | 0.32 (0.19, 0.49) | 0.19 (0.00, 0.46) |
| Proposed | CoCoLasso | 10 | 0.52 (0.28, 0.73) | 0.32 (0.00, 0.50) |
