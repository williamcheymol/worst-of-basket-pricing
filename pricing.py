"""
Worst-of basket note pricing via Monte Carlo.

A worst-of note pays a return linked to the WORST-PERFORMING asset in a
basket, rather than the average or any single asset. This module computes
its fair price and a 95% Monte Carlo confidence interval.
"""

import numpy as np

from simulation import S0, get_simulated_paths, N_PATHS, R, T

def worst_of_payoffs(terminal_values: np.ndarray, S0: np.ndarray):
    """
    Compute the worst-of performance for each simulated path.

    Parameters
    ----------
    terminal_values : np.ndarray, shape (n_paths, 3)
        Simulated terminal asset values from simulation.py.
    S0 : np.ndarray, shape (3,)
        Initial asset prices.

    Returns
    -------
    np.ndarray, shape (n_paths,)
        The worst-of performance min(perf_1, perf_2, perf_3) for each path.
    """
    # Compute each asset's performance Si(T)/Si(0), then take the minimum ACROSS ASSETS (axis=1) for each path.
    performances = terminal_values / S0
    worst_of = np.min(performances, axis=1)

    return worst_of


def price_worst_of_note(
    terminal_values: np.ndarray,
    S0: np.ndarray,
    notional: float,
    r: float,
    T: float,
):
    """
    Price a worst-of basket note via Monte Carlo and compute a 95%
    confidence interval on the price.

    Parameters
    ----------
    terminal_values : np.ndarray, shape (n_paths, 3)
        Simulated terminal asset values.
    S0 : np.ndarray, shape (3,)
        Initial asset prices.
    notional : float
        Notional amount of the note.
    r : float
        Risk-free rate.
    T : float
        Maturity in years.

    Returns
    -------
    (price, (ci_low, ci_high), payoffs) : (float, (float, float), np.ndarray)
        price is the Monte Carlo estimate of the note's fair value.
        (ci_low, ci_high) is the 95% confidence interval on that price.
        payoffs is the array of discounted dollar payoffs per path (for
        plotting / further analysis).
    """
    n_paths = terminal_values.shape[0]
    worst_of = worst_of_payoffs(terminal_values, S0)

    # Discount the risk-neutral expectation of the payoff:
    #   price = notional * e^{-rT} * mean(worst_of)
    price = np.exp(-r * T) * np.mean(worst_of) * notional

    # 95% Monte Carlo confidence interval on the price:
    #   half_width = 1.96 * notional * e^{-rT} * std(worst_of) / sqrt(n_paths)
    #   ci_low, ci_high = price - half_width, price + half_width
    half_width = 1.96 * notional * np.exp(-r*T) * np.std(worst_of) / np.sqrt(n_paths)
    ci_low, ci_high = price - half_width, price + half_width

    discounted_payoffs = notional * np.exp(-r * T) * worst_of
    return price, (ci_low, ci_high), discounted_payoffs


def get_note_price(
    terminal_values: np.ndarray = None,
    notional: float = 1_000_000.0,
):
    """
    Convenience wrapper using the module-level market parameters. If
    terminal_values is not provided, simulates a fresh set of paths.
    """
    if terminal_values is None:
        terminal_values = get_simulated_paths(N_PATHS)
    return price_worst_of_note(terminal_values, S0, notional, R, T)
