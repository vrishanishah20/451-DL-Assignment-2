# 451-DL-Assignment-2

# Portfolio Optimization: A Monte Carlo Study

### Author
Vrishani Shah  
MSAI 451 - Portfolio Optimization (Fall 2025)

---

## 🎯 Objective
This project explores portfolio optimization through **Monte Carlo simulation** to visualize the trade-off between **expected returns** and **risk (standard deviation)** when taking **long-only** and **short-allowed** positions across multiple assets.

---

## 🧩 Code Overview
The main program `monte_carlo_portfolio.py`:
- Simulates multivariate-normal asset returns
- Generates random portfolio weight combinations
- Computes portfolio mean returns and standard deviations
- Compares *short-allowed* vs *long-only* portfolios
- Saves results and plots the opportunity sets

---

## How to Run

### Default 4 Assets and synthetic data
```bash
python monte_carlo_portfolio.py --config default --samples 700 --outdir outputs
python monte_carlo_portfolio.py --config assets_alt.json --samples 1000 --outdir outputs_alt


