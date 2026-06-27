"""
Risk sensitivities for the worst-of basket note: per-asset Delta, and
price sensitivity to correlation (rho).
"""

import numpy as np

from correlation import get_correlation_setup
from simulation import S0, SIGMA, R, T, simulate_terminal_values, N_PATHS
from pricing import price_worst_of_note

def compute_delta_per_asset(
    notional: float = 1_000_000.0,
    bump: float = 0.01,
    rho_12: float = 0.85,
    rho_13: float = 0.75,
    rho_23: float = 0.70,
    n_paths: int = N_PATHS,
    seed: int = 42,
):
    """
    Compute the note's Delta with respect to each of the three assets'
    initial spot price, via bump-and-reprice.

    For each asset i: bump Si(0) by +1%, holding the OTHER two spots
    fixed, re-simulate with the SAME random seed (to isolate the effect
    of the bump from Monte Carlo noise), reprice, and take the
    difference from the base price.

    Parameters
    ----------
    notional : float
        Notional amount of the note.
    bump : float
        Relative bump size applied to each S0 in turn (e.g. 0.01 = +1%).
    rho_12, rho_13, rho_23 : float
        Basket correlations.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        Random seed, fixed across base and bumped simulations to reduce
        Monte Carlo noise in the *difference* (a form of variance
        reduction via common random numbers).

    Returns
    -------
    np.ndarray, shape (3,)
        [Delta_SP500, Delta_Nasdaq, Delta_EuroStoxx]
    """
    _, L = get_correlation_setup(rho_12, rho_13, rho_23)

    np.random.seed(seed)
    base_terminal = simulate_terminal_values(S0, SIGMA, R, T, L, n_paths)
    base_price, _, _ = price_worst_of_note(base_terminal, S0, notional, R, T)

    deltas = np.zeros(3)
    for i in range(3):
        bumped_S0 = S0.copy()
        bumped_S0[i] *= (1 + bump)
        np.random.seed(seed)
        bumped_terminal = simulate_terminal_values(bumped_S0, SIGMA, R, T, L, n_paths)
        bumped_price, _, _ = price_worst_of_note(bumped_terminal, S0, notional, R, T)
        deltas[i] = (bumped_price - base_price) / (S0[i] * bump)

    return deltas


def price_vs_rho(
    rho_values: np.ndarray,
    notional: float = 1_000_000.0,
    n_paths: int = N_PATHS,
    seed: int = 42,
):
    """
    Reprice the note across a range of (flat, equal) correlation values
    to show the price's monotonic dependence on rho.

    For simplicity, all three pairwise correlations (rho_12, rho_13,
    rho_23) are set to the SAME value at each point in rho_values.

    Parameters
    ----------
    rho_values : np.ndarray
        Array of correlation values to test, e.g. np.arange(0.3, 1.0, 0.05).
    notional : float
        Notional amount.
    n_paths : int
        Number of Monte Carlo paths per rho.
    seed : int
        Random seed, fixed across all rho values (common random numbers)
        to isolate the effect of rho from Monte Carlo noise.

    Returns
    -------
    np.ndarray, same shape as rho_values
        Note price at each tested correlation level.
    """
    prices = np.zeros_like(rho_values)

    for idx, rho in enumerate(rho_values):
        # Rebuild the correlation matrix and Cholesky
        # factor with rho_12 = rho_13 = rho_23 = rho, reset the random
        # seed (common random numbers across the loop), resimulate, and
        # reprice.
        _, L = get_correlation_setup(rho, rho, rho)
        np.random.seed(seed)
        terminal_values = simulate_terminal_values(S0, SIGMA, R, T, L, n_paths)
        price, _, _ = price_worst_of_note(terminal_values, S0, notional, R, T)
        prices[idx] = price

    return prices
