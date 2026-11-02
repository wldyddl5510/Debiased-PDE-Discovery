import numpy as np

from .loh_wainwright import LohWainwright


class CoCoLasso(LohWainwright):
    """Datta & Zou (2017), eq. (2.1), with (Gamma, gamma) as in Loh & Wainwright (2012), eq. (2.8):

    beta-hat = argmin beta'(Gamma)_+ beta/2 - gamma'beta + lam ||beta||_1,  Gamma = X'X/K - Sigma_W,  gamma = X'Y/K,
    where (Gamma)_+ is the nearest matrix >= eps I in the max norm, on the columns scaled to ||X_l||^2/K = 1 as the
    paper assumes. beta-hat is returned in the original units.

    lam is a number, or a grid searched by the corrected V-fold CV (5.4) over the z_i of a random design.
    """

    project = True

    def __init__(self, lam, **kwargs):
        super().__init__(lam, np.inf, standardize=True, **kwargs)
