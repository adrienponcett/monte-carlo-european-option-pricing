"""
Monte Carlo pricing of a European call option.

The Monte Carlo estimate is validated against the closed-form Black-Scholes
price, and the 1/sqrt(N) convergence rate of the Monte Carlo error is measured
empirically via a log-log regression.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def Monte_Carlo_price_call(S0, K, r, sigma, T, N, seed=69):
    """Price a European call by Monte Carlo under the risk-neutral measure.

    Parameters
    ----------
    S0, K, r, sigma, T : float
        Spot, strike, risk-free rate, volatility, maturity (in years).
    N : int
        Number of simulated terminal prices.
    seed : int
        Fixed seed for reproducible draws across runs.

    Returns
    -------
    price : float
        Discounted mean payoff (Monte Carlo estimate).
    se : float
        Standard error of the estimate (sigma_Y / sqrt(N)).
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(N)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    discounted = np.exp(-r * T) * np.maximum(ST - K, 0.0)
    price = discounted.mean()
    se = discounted.std(ddof=1) / np.sqrt(N)   # unbiased std -> standard error of the mean
    return price, se


def Black_Scholes_price_call(S0, K, r, sigma, T):
    """Closed-form Black-Scholes price of a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def plot_convergence(S0, K, r, sigma, T, N_values):
    """Plot the price convergence and the log-log decay of the Monte Carlo error.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The two-panel figure (price convergence + error decay).
    """
    prices, ses = [], []
    for N in N_values:
        price, se = Monte_Carlo_price_call(S0, K, r, sigma, T, int(N))
        prices.append(price)
        ses.append(se)

    prices = np.array(prices)
    errors = 1.96 * np.array(ses)              # half-width of the 95% confidence interval
    bs_price = Black_Scholes_price_call(S0, K, r, sigma, T)

    # Log-log regression to measure the convergence rate (expected slope: -0.5)
    slope, intercept = np.polyfit(np.log(N_values), np.log(errors), 1)
    fit_line = np.exp(intercept) * N_values.astype(float) ** slope

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: price convergence with 95% CI error bars
    ax1.errorbar(N_values, prices, yerr=errors,
                 fmt='o', capsize=4, label='Monte Carlo (95% CI)')
    ax1.axhline(bs_price, color='red', linestyle='--', label='Black-Scholes (exact)')
    ax1.set_xscale('log')                      # N spans several orders of magnitude
    ax1.set_xlabel('Number of simulations N')
    ax1.set_ylabel('Call price')
    ax1.set_title('Price convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: error decay in log-log (a power law appears as a straight line)
    ax2.loglog(N_values, errors, 'o', label='MC error (95% CI half-width)')
    ax2.loglog(N_values, fit_line, '--', label=f'Fit: slope = {slope:.3f}')
    ax2.set_xlabel('Number of simulations N')
    ax2.set_ylabel('Error (95% CI half-width)')
    ax2.set_title('Error decay - convergence rate')
    ax2.legend()
    ax2.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 100, 0.05, 0.2, 1

    # Sanity check: Monte Carlo vs closed-form Black-Scholes
    price, se = Monte_Carlo_price_call(S0, K, r, sigma, T, 1_000_000)
    bs = Black_Scholes_price_call(S0, K, r, sigma, T)
    print(f"Monte Carlo  : {price:.4f} +/- {1.96 * se:.4f}  (95% CI)")
    print(f"Black-Scholes: {bs:.4f}")

    # Convergence study
    N_values = np.logspace(2, 7, 12).astype(int)
    fig = plot_convergence(S0, K, r, sigma, T, N_values)
    fig.savefig("convergence.png", dpi=150)
    plt.show()