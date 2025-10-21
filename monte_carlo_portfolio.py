#!/usr/bin/env python3
"""
Portfolio Optimization: A Monte Carlo Study (Python version)

Replicates the R jump-start code provided by Tom Miller (June 2025).

This script:
  • Generates simulated multivariate-normal returns for 4 assets
  • Samples random portfolio weights (long-only vs shorts OK)
  • Computes portfolio mean & standard-deviation
  • Produces CSV outputs and scatter-plot figures

Usage:
    python monte_carlo_portfolio.py --config default --samples 700 --seed 1111 --outdir outputs
    python monte_carlo_portfolio.py --config assets_alt.json --samples 1000 --seed 42 --outdir outputs_alt
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def load_config(config_arg: str):
    """Load default or custom asset configuration."""
    if config_arg == "default":
        names = ["A", "B", "C", "D"]
        means = [0.02, 0.07, 0.15, 0.20]
        sds   = [0.05, 0.12, 0.17, 0.25]
        cor = [
            [1.0, 0.3, 0.3, 0.3],
            [0.3, 1.0, 0.6, 0.6],
            [0.3, 0.6, 1.0, 0.6],
            [0.3, 0.6, 0.6, 1.0]
        ]
    else:
        with open(config_arg, "r") as f:
            cfg = json.load(f)
        names = cfg["asset_names"]
        means = cfg["means"]
        sds   = cfg["sds"]
        cor   = cfg["cor"]
    return names, np.array(means), np.array(sds), np.array(cor)


def cov_from_sd_corr(sds, corr):
    """Compute covariance matrix from SDs and correlations."""
    D = np.diag(sds)
    return D @ corr @ D


def make_weights(n_assets, shorts_ok, rng):
    """Generate one random weight vector."""
    if shorts_ok:
        w_partial = rng.uniform(-1.0, 1.0, n_assets - 1)
        last = 1.0 - np.sum(w_partial)
        w = np.append(w_partial, last)
    else:
        w = rng.uniform(0.0, 1.0, n_assets)
        w /= np.sum(w)
    return w


def simulate_returns(samples, mean, cov, rng):
    """Generate simulated multivariate-normal asset returns."""
    return rng.multivariate_normal(mean, cov, size=samples)


def evaluate_portfolios(returns, cov_emp, weights):
    """Compute mean & SD for each portfolio weight vector."""
    results = []
    for w in weights:
        pr = returns @ w
        mean_r = pr.mean()
        sd_r = float(np.sqrt(w.T @ cov_emp @ w))
        pos = 2 if np.all(w >= 0) else 1
        results.append((mean_r, sd_r, pos))
    df = pd.DataFrame(results, columns=["returnMean", "returnSD", "Positions"])
    return df


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------
def run(config_arg, seed, samples, outdir):
    names, means, sds, corr = load_config(config_arg)
    n_assets = len(names)
    cov = cov_from_sd_corr(sds, corr)
    Path(outdir).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    returns = simulate_returns(samples, means, cov, rng)
    df_returns = pd.DataFrame(returns, columns=names)
    cov_emp = np.cov(df_returns, rowvar=False)

    # ---- Shorts OK ----
    rng_w = np.random.default_rng(seed + 999)
    weights_shorts = np.vstack([make_weights(n_assets, True, rng_w) for _ in range(samples)])
    df_shorts = evaluate_portfolios(df_returns.to_numpy(), cov_emp, weights_shorts)
    df_shorts["ShortsOK"] = "Shorts OK"

    # ---- Long-only ----
    weights_long = np.vstack([make_weights(n_assets, False, rng_w) for _ in range(samples)])
    df_long = evaluate_portfolios(df_returns.to_numpy(), cov_emp, weights_long)
    df_long["ShortsOK"] = "Long Positions Only"

    # ---- Save outputs ----
    df_returns.to_csv(f"{outdir}/simulated_returns.csv", index=False)
    df_shorts.to_csv(f"{outdir}/portfolios_shorts_ok.csv", index=False)
    df_long.to_csv(f"{outdir}/portfolios_long_only.csv", index=False)

    combined = pd.concat([df_shorts, df_long], ignore_index=True)
    combined.to_csv(f"{outdir}/portfolio_results_combined.csv", index=False)

    # ---- Plot ----
    for label, df in [("Shorts OK", df_shorts), ("Long Positions Only", df_long)]:
        plt.figure()
        plt.scatter(df["returnSD"], df["returnMean"], s=8)
        plt.xlabel("Risk: Standard Deviation of Portfolio Returns")
        plt.ylabel("Return: Mean of Portfolio Returns")
        plt.title(f"Opportunity Set – {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/opportunity_{label.replace(' ', '_').lower()}.png", dpi=200)
        plt.close()

    print(f"✅ Run complete. Results saved in '{outdir}'.")


# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="default", help='Either "default" or path to JSON file')
    ap.add_argument("--samples", type=int, default=700)
    ap.add_argument("--seed", type=int, default=1111)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    run(args.config, args.seed, args.samples, args.outdir)
