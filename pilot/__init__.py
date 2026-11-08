"""Pilot estimates u-hat of u_*. Each one is pilot(data, seed) -> u-hat, a callable u-hat(*z) of the shape of u_*(*z).

main.run splits the data first and passes only the pilot half here, so u-hat and beta-hat come from disjoint samples.
"""
from .perturbed_truth import perturbed_truth
