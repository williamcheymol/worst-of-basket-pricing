"""
Correlation matrix construction and Cholesky decomposition for a 3-asset
basket (S&P 500, Nasdaq, Euro Stoxx).

This module builds the correlation matrix between the three underlyings
and decomposes it so we can later simulate CORRELATED Brownian motions
from INDEPENDENT random draws.
"""

import numpy as np


RHO_12 = 0.85  # correlation between S&P 500 and Nasdaq
RHO_13 = 0.75  # correlation between S&P 500 and Euro Stoxx
RHO_23 = 0.70  # correlation between Nasdaq and Euro Stoxx


def build_correlation_matrix(rho_12: float, rho_13: float, rho_23: float):
    """
    Build the 3x3 correlation matrix for (S&P 500, Nasdaq, Euro Stoxx).

    Parameters
    ----------
    rho_12, rho_13, rho_23 : float
        Pairwise correlations between assets 1-2, 1-3, and 2-3.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Symmetric correlation matrix with unit diagonal.
    """
    # Assemble the matrix:
    corr_matrix = np.array( [[1, rho_12, rho_13], 
                             [rho_12, 1, rho_23], 
                             [rho_13, rho_23, 1]])

    return corr_matrix


def check_positive_definite(corr_matrix: np.ndarray):
    """
    Assert that the correlation matrix is positive definite by checking
    that all of its eigenvalues are strictly positive.

    Parameters
    ----------
    corr_matrix : np.ndarray
        The correlation matrix to check.

    Raises
    ------
    AssertionError
        If any eigenvalue is non-positive (matrix is not valid).
    """
    # Compute eigenvalues and assert they are all > 0.
    eigenvalues = np.linalg.eigvals(corr_matrix)

    assert np.all(eigenvalues > 0), (
        f"Correlation matrix is not positive definite: eigenvalues = {eigenvalues}"
    )


def cholesky_decomposition(corr_matrix: np.ndarray):
    """
    Compute the Cholesky factor L such that corr_matrix = L @ L.T.

    Parameters
    ----------
    corr_matrix : np.ndarray
        A symmetric positive definite correlation matrix.

    Returns
    -------
    np.ndarray, shape (3, 3)
        Lower-triangular matrix L.
    """
    # Cholesky decomposes a SPD matrix A into L @ L.T.
    # This allows us to correlate independent Brownian motions later by
    # computing W = Z @ L.T.
    L = np.linalg.cholesky(corr_matrix)

    return L


def get_correlation_setup(rho_12: float = RHO_12, rho_13: float = RHO_13,
                           rho_23: float = RHO_23):
    """
    Build the correlation matrix, validate it, and compute its Cholesky
    factor in one call.

    Returns
    -------
    (corr_matrix, L) : (np.ndarray, np.ndarray)
    """
    corr_matrix = build_correlation_matrix(rho_12, rho_13, rho_23)
    check_positive_definite(corr_matrix)
    L = cholesky_decomposition(corr_matrix)
    return corr_matrix, L
