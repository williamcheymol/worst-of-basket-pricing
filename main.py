"""
Entry point: simulates the 3-asset basket, prices the worst-of note,
computes sensitivities, prints a summary, and produces diagnostic plots.
"""

import numpy as np
import matplotlib.pyplot as plt

from correlation import get_correlation_setup, RHO_12, RHO_13, RHO_23
from simulation import S0, SIGMA, R, T, N_PATHS, simulate_terminal_values
from pricing import price_worst_of_note
from sensitivities import compute_delta_per_asset, price_vs_rho
from style import COLORS, apply_theme

apply_theme()

np.random.seed(42)

NOTIONAL = 1_000_000.0
ASSET_NAMES = ["S&P 500", "Nasdaq", "Euro Stoxx"]


def main():
    _, L = get_correlation_setup(RHO_12, RHO_13, RHO_23)
    terminal_values = simulate_terminal_values(S0, SIGMA, R, T, L, N_PATHS)

    price, (ci_low, ci_high), payoffs = price_worst_of_note(
        terminal_values, S0, NOTIONAL, R, T
    )

    deltas = compute_delta_per_asset(notional=NOTIONAL)

    print(f"Note price:      ${price:,.0f}")
    print(f"95% CI:          [${ci_low:,.0f}, ${ci_high:,.0f}]")
    print(f"Delta S&P:       {deltas[0]:.2f}")
    print(f"Delta Nasdaq:    {deltas[1]:.2f}")
    print(f"Delta EuroStoxx: {deltas[2]:.2f}")

    plot_worst_of_distribution(terminal_values)
    plot_price_vs_rho(price)
    plot_delta_per_asset(deltas)

    plt.show()


def plot_worst_of_distribution(terminal_values: np.ndarray):
    """Histogram of worst-of performances across all simulated paths."""
    performances = terminal_values / S0
    worst_of = np.min(performances, axis=1)

    mean_perf = np.mean(worst_of)
    p5 = np.percentile(worst_of, 5)

    plt.figure(figsize=(8, 5))
    plt.hist(worst_of, bins=80, color=COLORS["blue"], alpha=0.85)
    plt.axvline(mean_perf, color=COLORS["yellow"], linestyle="--",
                label=f"Mean = {mean_perf:.3f}")
    plt.axvline(p5, color=COLORS["red"], linestyle="--",
                label=f"5th percentile = {p5:.3f}")
    plt.xlabel("Worst-of performance (min of 3 assets)")
    plt.ylabel("Number of paths")
    plt.title("Distribution of Worst-Of Performance (10,000 paths)",
              color=COLORS["white"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graphiques/Distribution.png", dpi=150)


def plot_price_vs_rho(base_price: float):
    """Note price as a function of correlation rho (flat across pairs)."""
    rho_values = np.arange(0.30, 0.96, 0.05)
    prices = price_vs_rho(rho_values, notional=NOTIONAL)

    base_rho = (RHO_12 + RHO_13 + RHO_23) / 3

    plt.figure(figsize=(8, 5))
    plt.plot(rho_values, prices, color=COLORS["pink"], linewidth=2)
    plt.axvline(base_rho, color=COLORS["yellow"], linestyle="--",
                label=f"Base case avg rho = {base_rho:.2f}")
    plt.xlabel("Correlation rho (flat across all pairs)")
    plt.ylabel("Note price ($)")
    plt.title("Worst-Of Note Price vs Correlation", color=COLORS["white"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Graphiques/PriceVsRho.png", dpi=150)


def plot_delta_per_asset(deltas: np.ndarray):
    """Horizontal bar chart of per-asset delta."""
    plt.figure(figsize=(8, 5))
    plt.barh(ASSET_NAMES, deltas, color=COLORS["yellow"])
    plt.xlabel("Delta (price change in USD per USD 1 move in S0)")
    plt.title("Delta per Asset", color=COLORS["white"])
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("Graphiques/Delta.png", dpi=150)


if __name__ == "__main__":
    main()
