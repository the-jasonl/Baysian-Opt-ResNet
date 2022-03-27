import numpy as np

import matplotlib.pyplot as plt


def plot_approximation(gpr, X, X_sample, Y_sample, X_next=None, show_legend=False):
    """_summary_

    Args:
        gpr (_type_): _description_
        X (_type_): _description_
        X_sample (_type_): _description_
        Y_sample (_type_): _description_
        X_next (_type_, optional): _description_. Defaults to None.
        show_legend (bool, optional): _description_. Defaults to False.
    """
    # surrogate resolution
    mu, std = gpr.predict(X, return_std=True)
    plt.fill_between(X.ravel(),
                     mu.ravel() + 1.645 * std,
                     mu.ravel() - 1.645 * std,
                     alpha=0.4)
    plt.plot(X, mu, 'b-', lw=1, label='Surrogate function')
    plt.plot(X_sample, Y_sample, 'kx', mew=3, label='Noisy samples')
    if X_next:
        plt.axvline(x=X_next, ls='--', c='k', lw=1)
    if show_legend:
        plt.legend()


def plot_acquisition(X, Y, X_next, show_legend=False):
    plt.plot(X, Y, 'r-', lw=1, label='Acquisition function')
    plt.axvline(x=X_next, ls='--', c='k', lw=1, label='Next sampling location')
    if show_legend:
        plt.legend()
