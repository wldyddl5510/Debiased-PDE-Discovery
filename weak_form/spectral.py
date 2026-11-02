import numpy as np

from .weak_form import TestFunctions


def _critical_modes(data, alpha):
    # k* per component and coordinate, with N = n^(1/d) modes per coordinate. |sum_i U_i exp(-2 pi i <m, z_i/L>)|^2 / sum_i U_i^2
    # has mean 1 plus the signal at the mode m, whether the floor comes from the z_i or from the noise. A mode above log(#modes/alpha)
    # is signal, and k* is the last one counted up from 1, stopping at two empty modes in a row
    L = [hi - lo for lo, hi in data.domain]
    d = len(L)
    N = round(data.U.shape[-1]**(1/d))
    modes = np.arange(-(N//2), N - N//2)
    factors = [x for a, (z, l) in enumerate(zip(data.coords, L)) for x in (np.exp(-2j*np.pi*np.outer(modes, z/l)), [a, d])]
    ks = np.zeros((len(data.U), d))
    for c, u in enumerate(data.U):
        power = np.abs(np.einsum(*factors, u, [d], list(range(d)), optimize=True))**2/np.sum(u**2)
        hit = power > np.log(power.size/alpha)
        for a in range(d):
            found = set(np.abs(modes)[hit.any(axis=tuple(b for b in range(d) if b != a))])
            k = 0
            while k + 1 in found or k + 2 in found:
                k += 1 if k + 1 in found else 2
            ks[c, a] = max(k, 1)
    return ks, N


def spectral_test_functions(data, equations, width=1.0, spacing=50, alpha=0.1):
    """Test functions for a random design, after Appendix A of Messenger & Bortz (2021), whose spectrum and support need a grid:

    p_a = the highest derivative order in coordinate a + 1,
    R_a = sigma_a sqrt(2 p_a + 3) <= L_a/2 - L_a/N,  sigma_a = width L_a/(2 pi k*_a), averaged over the components,
    with the centers on a lattice of spacing L_a/spacing in [lo_a + R_a, hi_a - R_a], which does not depend on the data.
    """
    if data.design != "random":
        raise ValueError("spectral_test_functions needs a random design")
    terms = [term for equation in equations for term in (equation.lhs, *equation.rhs)]
    ks, N = _critical_modes(data, alpha)
    centers, radii, degrees = [], [], []
    for a, (lo, hi) in enumerate(data.domain):
        L = hi - lo
        degrees.append(max(term.derivative[a] for term in terms) + 1)
        radii.append(min(np.mean(width*L/(2*np.pi*ks[:, a]))*np.sqrt(2*degrees[a] + 3), L/2 - L/N))
        centers.append(lo + radii[a] + np.arange(int((L - 2*radii[a])/(L/spacing)) + 1)*L/spacing)
    return TestFunctions(tuple(centers), tuple(radii), tuple(degrees))
