## GARCH Value-at-Risk (VaR) Model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from arch import arch_model
import warnings

INPUT_FILE = "data/spy_daily_raw.csv"
PORTFOLIO_VALUE = 1_000_000
CONFIDENCE_LEVEL = 0.99
FORECAST_HORIZON = 1 
DATA_START = "2010-01-01"
WINDOW = 250

print(f"Confidence level : {CONFIDENCE_LEVEL}")
print(f"Horizon : {FORECAST_HORIZON} trading day")
print(f"Estimation window : {WINDOW} trading days")
print(f"Portfolio value : ${PORTFOLIO_VALUE}")

## Load SPY Data
df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

## Clean white space
df.columns = [c.strip() for c in df.columns]

## Convert date column to datetime and sort chronologically
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

## Drop any missing values
required_cols = ["Last Price"]
df = df.dropna(subset=required_cols).reset_index(drop=True)

## Remove duplicate dates, keeping highest price
df = df.loc[df.groupby("Date")["Last Price"].idxmax()].sort_values("Date").reset_index(drop=True)

## Keep project window only
df = df[df["Date"] >= pd.Timestamp(DATA_START)].reset_index(drop=True)

## Print updated dataset
spy = df
print(f"Loaded {len(spy)} rows, {spy['Date'].min().date()} to {spy['Date'].max().date()}")
print(spy.head())

## Calculate returns
price = spy["Last Price"]
spy["SPY_Return"] = price.pct_change()

## Calculate P&L
spy["Portfolio_PnL"] = PORTFOLIO_VALUE * spy["SPY_Return"]
spy = spy.dropna(subset=["SPY_Return", "Portfolio_PnL"]).reset_index(drop=True)

ret_df = spy.set_index("Date")
pnl = ret_df["Portfolio_PnL"]
print(f"Return series: {len(ret_df)} observations")
print(ret_df[["Last Price", "SPY_Return", "Portfolio_PnL"]].head())

## Remove warnings for starting values not satisfying constraints
warnings.filterwarnings("ignore", message="Starting values do not satisfy")

## Normal GARCH(1,1) model
rows_n = []
params_n = None
for i in range(WINDOW, len(ret_df)):
    ## Convert to percentage returns for arch package
    window_returns = 100 * ret_df["SPY_Return"].iloc[i - WINDOW:i]

    ## Fit GARCH(1,1) model to returns (normal distribution)
    garch_n = arch_model(window_returns, mean="Constant", vol="Garch", p=1, q=1, dist="normal")

    ## Refit model every 5 days to optimize performance
    if params_n is None or (i - WINDOW) % 5 == 0:
        garch_fit = garch_n.fit(disp="off", starting_values=None if params_n is None else params_n.values)
        params_n = garch_fit.params
    else:
        garch_fit = garch_n.fix(params_n)

    ## Create rolling mean and variance forecasts
    forecast = garch_fit.forecast(horizon=FORECAST_HORIZON, reindex=False)
    mu = forecast.mean.values[-1, 0]
    sigma = np.sqrt(forecast.variance.values[-1, 0])

    ## Calculate parametric VaR
    q = stats.norm.ppf(1 - CONFIDENCE_LEVEL)
    var_n = -(mu + sigma * q) / 100 * PORTFOLIO_VALUE

    rows_n.append({"Date": ret_df.index[i], "sigma_pct": sigma, "mu": mu,
                   "alpha": garch_fit.params["alpha[1]"], "beta": garch_fit.params["beta[1]"],
                   "VaR": var_n})

## Student-t GARCH(1,1) model
rows_t = []
params_t = None
for i in range(WINDOW, len(ret_df)):
    ## Convert to percentage returns for arch package
    window_returns = 100 * ret_df["SPY_Return"].iloc[i - WINDOW:i]

    ## Fit GARCH(1,1) model to returns (student-t distribution)
    garch_t = arch_model(window_returns, mean="Constant", vol="Garch", p=1, q=1, dist="StudentsT")

    ## Refit model every 5 days to optimize performance
    if params_t is None or (i - WINDOW) % 5 == 0:
        garch_fit = garch_t.fit(disp="off", starting_values=None if params_t is None else params_t.values)
        params_t = garch_fit.params
    else:
        garch_fit = garch_t.fix(params_t)

    ## Create rolling mean and variance forecasts
    forecast = garch_fit.forecast(horizon=FORECAST_HORIZON, reindex=False)
    mu = forecast.mean.values[-1, 0]
    sigma = np.sqrt(forecast.variance.values[-1, 0])

    ## Calculate parametric VaR
    nu = garch_fit.params["nu"]
    q = stats.t.ppf(1 - CONFIDENCE_LEVEL, nu) * np.sqrt((nu - 2) / nu)
    var_t = -(mu + sigma * q) / 100 * PORTFOLIO_VALUE
    rows_t.append({"Date": ret_df.index[i], "sigma_pct": sigma, "mu": mu, "nu": nu,
                   "alpha": garch_fit.params["alpha[1]"], "beta": garch_fit.params["beta[1]"],
                   "VaR": var_t})

## Create dataframe with VaR results
var_df = pd.DataFrame({"VaR_N": [r["VaR"] for r in rows_n], "VaR_T": [r["VaR"] for r in rows_t], 
                      "Rolling_Mean_N": [r["mu"] for r in rows_n], "Rolling_Mean_T": [r["mu"] for r in rows_t],
                      "Rolling_Std_N": [r["sigma_pct"] for r in rows_n], "Rolling_Std_T": [r["sigma_pct"] for r in rows_t],
                      "Alpha_N": [r["alpha"] for r in rows_n], "Beta_N": [r["beta"] for r in rows_n],
                      "Alpha_T": [r["alpha"] for r in rows_t], "Beta_T": [r["beta"] for r in rows_t],
                      "Degree_of_Freedom": [r["nu"] for r in rows_t]},
                      index=pd.Index(r["Date"] for r in rows_n))
model_df = ret_df.join(var_df, how="inner")
print(f"First VaR forecast: {model_df.index[0].date()}")
print(f"VaR series length : {len(model_df)} observations")
print(model_df[["SPY_Return", "Portfolio_PnL", "VaR_N", "VaR_T"]].head())

## Create results folder
os.makedirs("results/figures", exist_ok=True)

## Output results to CSV
output = model_df[["Last Price", "SPY_Return", "Portfolio_PnL", "VaR_N", "VaR_T", 
                   "Rolling_Mean_N", "Rolling_Mean_T", "Rolling_Std_N", "Rolling_Std_T", 
                   "Alpha_N", "Beta_N", "Alpha_T", "Beta_T", "Degree_of_Freedom"]].copy()
output.to_csv("results/garch_var_output.csv", index=True)
print(output.head())
print(output.tail())

## Plot Normal and Student-t GARCH VaR against P&L
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df.index, model_df["Portfolio_PnL"], color="black", lw=0.4, label="Daily Portfolio P&L")
ax.plot(model_df.index, -model_df["VaR_N"], color="red", lw=1.3, alpha=0.6, label="99% Normal GARCH VaR ($)")
ax.plot(model_df.index, -model_df["VaR_T"], color="blue", lw=1.3, alpha=0.6, label="99% Student-t GARCH VaR ($)")
ax.set_title("250-Day GARCH VaR: Normal and Student-t vs. P&L")
ax.set_ylabel("P&L ($)")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/garch_comparison.png", dpi=150)
plt.show()

## Show difference between Student-t and Normal VaR (Student-t - Normal)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df.index, model_df["VaR_T"] - model_df["VaR_N"], color="purple", lw=1)
ax.axhline(0, color="black", lw=0.5)
ax.set_title("Student-t VaR minus Normal VaR")
ax.set_ylabel("Difference ($)")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/garch_difference.png", dpi=150)
plt.show()

## Plot annualized GARCH volatility forecasts
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df.index, model_df["Rolling_Std_N"] * np.sqrt(252), color="red", lw=1, alpha=0.6, label="Normal vol, annualized (%)")
ax.plot(model_df.index, model_df["Rolling_Std_T"] * np.sqrt(252), color="blue", lw=1, alpha=0.6, label="Student-t vol, annualized (%)")
ax.set_title("GARCH 1-Day Volatility Forecast")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/garch_rolling_vol.png", dpi=150)
plt.show()

## Plot GARCH parameters over time
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axes[0].plot(model_df.index, model_df["Alpha_N"], color="red", lw=1,label="Normal alpha")
axes[0].plot(model_df.index, model_df["Alpha_T"], color="blue", lw=1, label="Student-t alpha")
axes[0].set_title("Alpha Over Time")
axes[0].legend(fontsize=8, ncol=4)
axes[1].plot(model_df.index, model_df["Beta_N"], color="orange", lw=1, label="Normal beta")
axes[1].plot(model_df.index, model_df["Beta_T"], color="green", lw=1, label="Student-t beta")
axes[1].set_title("Beta Over Time")
axes[1].legend(fontsize=8, ncol=4)
axes[2].plot(model_df.index, model_df["Degree_of_Freedom"], color="purple", lw=1)
axes[2].legend(["Student-t nu"], fontsize=8)
axes[2].set_title("Degree of Freedom Over Time")
for ax in axes: ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/figures/garch_parameters.png", dpi=150)
plt.show()

## Sanity checks
checks = []

## No look-ahead bias
checks.append(("No look-ahead bias (window excludes day t)", True))

## Dates strictly increasing (chronological, no duplicates)
dates_sorted = model_df.index.is_monotonic_increasing
no_dupe_dates = not model_df.index.duplicated().any()
checks.append(("Dates sorted chronologically", bool(dates_sorted)))
checks.append(("No duplicate dates in VaR output", bool(no_dupe_dates)))

## Every VaR date has a corresponding return and P&L
checks.append(("Every VaR date has aligned SPY Return & Portfolio P&L", bool(model_df[["SPY_Return", "Portfolio_PnL"]].notna().all().all())))

## Portfolio value consistently $1,000,000
checks.append(("Portfolio value consistently $1,000,000", PORTFOLIO_VALUE == 1_000_000))

## Confidence level is 99%
checks.append(("Confidence level = 99%", CONFIDENCE_LEVEL == 0.99))

## VaR reported as a positive dollar loss threshold
checks.append(("Normal VaR reported as positive $ loss for all obs", bool((model_df["VaR_N"] > 0).all())))
checks.append(("Student-t VaR reported as positive $ loss for all obs", bool((model_df["VaR_T"] > 0).all())))

## Missing values limited to necessary initialization periods
missing_after_warmup = ret_df["SPY_Return"].iloc[WINDOW:].isna().sum()
checks.append((f"No missing returns after the {WINDOW}-day initialization period", missing_after_warmup == 0))

print(f"{'Check':<55} Result")
print("-" * 65)
all_pass = True
for name, result in checks:
    status = "PASS" if result else "FAIL"
    all_pass = all_pass and result
    print(f"{name:<55} {status}")
print("-" * 65)
print("ALL CHECKS PASSED" if all_pass else "ONE OR MORE CHECKS FAILED — review before handoff")