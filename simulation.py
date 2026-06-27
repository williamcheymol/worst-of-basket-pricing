"""
Monte Carlo simulation of correlated terminal asset values under
Geometric Brownian Motion (GBM), for the 3-asset basket
(S&P 500, Nasdaq, Euro Stoxx).
"""

import numpy as np

from correlation import get_correlation_setup


S0 = np.array([100.0, 100.0, 100.0])          # initial prices: S&P, Nasdaq, EuroStoxx
SIGMA = np.array([0.20, 0.25, 0.22])          # volatilities: S&P, Nasdaq, EuroStoxx
R = 0.02                                      # risk-free rate
T = 3.0                                       # maturity (years)
N_PATHS = 10_000                              # number of Monte Carlo paths


def simulate_terminal_values(
    S0: np.ndarray,
    sigma: np.ndarray,
    r: float,
    T: float,
    L: np.ndarray,
    n_paths: int,
):
    """
    Simulate correlated terminal asset values under risk-neutral GBM.

    Parameters
    ----------
    S0 : np.ndarray, shape (3,)
        Initial asset prices.
    sigma : np.ndarray, shape (3,)
        Asset volatilities.
    r : float
        Risk-free rate (used as the risk-neutral drift for every asset).
    T : float
        Maturity in years.
    L : np.ndarray, shape (3, 3)
        Cholesky factor of the correlation matrix (from correlation.py).
    n_paths : int
        Number of Monte Carlo paths to simulate.

    Returns
    -------
    np.ndarray, shape (n_paths, 3)
        Simulated terminal values Si(T) for each path and each asset.
    """
    # Draw independent standard normal noise.
    Z = np.random.normal(0, 1, size=(n_paths, 3))

    # Correlate the noise using the Cholesky factor.
    # W = Z @ L.T introduces the target correlation structure.
    W = Z @ L.T

    # Apply the GBM closed-form terminal value formula to
    # every path and every asset at once (vectorized, no loop needed).
    terminal_values = S0 * np.exp((r - sigma**2/2)*T + sigma* np. sqrt(T)*W)

    return terminal_values


def get_simulated_paths(
    n_paths: int = N_PATHS,
    rho_12: float = 0.85,
    rho_13: float = 0.75,
    rho_23: float = 0.70,
):
    """
    Convenience wrapper: build the correlation structure and simulate
    terminal values in one call, using the module-level market parameters.

    Returns
    -------
    np.ndarray, shape (n_paths, 3)
    """
    _, L = get_correlation_setup(rho_12, rho_13, rho_23)
    return simulate_terminal_values(S0, SIGMA, R, T, L, n_paths)
