import matplotlib.pyplot as plt
import numpy as np
from typing import List

from gaussian_utils.gaussian import GaussianProcessOptimizer


def plot_approximation(gpr: GaussianProcessOptimizer, X: np.ndarray, X_sample: List[float],
                       Y_sample: List[float], X_next: float, show_legend: bool = False):
    """Plots the observations, posterior mean and uncertainty estimate

    Args:
        gpr (GaussianProcessOptimizer): GaussianProcessOptimizer
        X (np.ndarray): X values for posterior mean
        X_sample (List[float]): Sampled X
        Y_sample (List[float]): Sampled Y
        X_next (float): Next sampling location
        show_legend (bool, optional): Plot legend. Defaults to False.
    """
    # surrogate function
    mu, std = gpr.predict(X, return_std=True)
    # uncertainty estimate
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.645 * std,
                     mu.ravel() - 1.645 * std,
                     alpha=0.4,
                     label="Uncertainty estimate")
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Sampled X')
    plt.axvline(x=X_next, ls='--', c='k', lw=1)
    ax = plt.gca()
    ax.set_xlabel('learning rate')
    ax.set_ylabel('test loss')

    if show_legend:
        ax.set_ylim(ymin=0)
        plt.legend()


def plot_acquisition(X: np.ndarray, Y: np.ndarray, X_next: float, show_legend: bool = False):
    """Plots the acquisition function and the proposed next location

    Args:
        X (np.ndarray): X values for acquisition function
        Y (np.ndarray): Y values for acquisition function
        X_next (float): Proposed next sampling location
        show_legend (bool, optional): Plot legend. Defaults to False.
    """
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    ax = plt.gca()
    ax.set_xlabel('learning rate')
    ax.set_ylabel('expected improvement')
    if show_legend:
        plt.legend()
