"""Simulated oracle pilot: the true solution plus a smooth random perturbation of size n^(-gamma)."""
import numpy as np


def perturbed_truth(data, pde, zeta=1.0, gamma=1/3, modes=8, seed=None):
    """u-hat_c = u_*c + zeta n^(-gamma) g_c, with n the size of the pilot sample, as in nls_debiasing.ipynb.

    g_c = sum_k a_ck prod_a cos(pi k_a (z_a - lo_a)/L_a) over k in {0, ..., modes}^d, with a_ck ~ N(0, 1) scaled so
    that g_c^2 has mean 1 over the domain. The notebook adds +-1 at each z_i instead, which is not a function of z.
    """
    rng = np.random.default_rng(seed)
    d, k = len(pde.domain), np.arange(modes + 1)
    a = rng.standard_normal((len(data.U),) + (modes + 1,)*d)
    # On [0, 1], cos(pi k s) has mean square 1 at k = 0 and 1/2 otherwise, and the products are orthogonal
    w = np.prod(np.meshgrid(*[np.where(k == 0, 1., .5)]*d, indexing="ij"), axis=0)
    a = zeta*data.U[0].size**(-gamma)*a/np.sqrt(np.sum(a**2*w, axis=tuple(range(1, d + 1)), keepdims=True))

    def uhat(*z):
        index = list(range(d + 1, d + 1 + np.ndim(z[0])))
        factors = [x for i, (zi, (lo, hi)) in enumerate(zip(z, pde.domain))
                   for x in (np.cos(np.pi*np.multiply.outer(k, (zi - lo)/(hi - lo))), [1 + i, *index])]
        return pde.solution(*z) + np.einsum(a, list(range(d + 1)), *factors, [0, *index], optimize=True)

    return uhat
