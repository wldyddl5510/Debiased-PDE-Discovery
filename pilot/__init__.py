"""Pilot estimates u-hat of u_*. Each one is pilot(data, seed) -> u-hat, a callable u-hat(*z) of the shape of u_*(*z).

main.run passes all n observations here; WeakForm then evaluates u-hat at m new points (random or grid).
"""
from .perturbed_truth import perturbed_truth
from .pinn import PINN
