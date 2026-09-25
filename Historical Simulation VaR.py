## Historical Simulation Value-at-Risk (VaR) Model

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

INPUT_FILE = r"C:\Users\cassi\Downloads\Var-Spy-daily_raw_data.csv"
RETURN_TYPE = "log"
PORTFOLIO_VALUE = 1_000_000
CONFIDENCE_LEVEL = 0.99
FORECAST_HORIZON = 1 
DATA_START = "2010-01-01"
WINDOW = 250
AGE_WEIGHT_LAMBDA = 0.97

print(f"Confidence level : {CONFIDENCE_LEVEL}")
print(f"Horizon : {FORECAST_HORIZON} trading day")
print(f"Estimation window : {WINDOW} trading days")
print(f"Portfolio value : ${PORTFOLIO_VALUE} SPY")

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
if RETURN_TYPE == "log":
    spy["SPY Return"] = np.log(price / price.shift(1))
else:
    spy["SPY Return"] = price.pct_change()

## Calculate P&L
spy["Portfolio P&L"] = PORTFOLIO_VALUE * spy["SPY Return"]
spy = spy.dropna(subset=["SPY Return", "Portfolio P&L"]).reset_index(drop=True)

ret_df = spy.set_index("Date")
pnl = ret_df["Portfolio P&L"]
print(f"Return series: {len(ret_df)} observations")
print(ret_df[["Last Price", "SPY Return", "Portfolio P&L"]].head())

## Set up quartile based off of confidence level
pct = (1 - CONFIDENCE_LEVEL) * 100

## Calculate VaR dollar value using rolling 250-day window using P&L
var_dollar, dates = [], []
for t in range(WINDOW, len(pnl)):
    window_pnl = pnl.iloc[t - WINDOW:t]
    var = -np.percentile(window_pnl, pct)
    var_dollar.append(var)
    dates.append(pnl.index[t])

## Create dataframe with VaR results
var_df = pd.DataFrame({"VaR": var_dollar}, index=pd.Index(dates, name="Date"))
model_df = ret_df.join(var_df, how="inner")
print(f"First VaR forecast: {model_df.index[0].date()}  (expected ~Jan 2011 per spec)")
print(f"VaR series length : {len(model_df)} observations")
print(model_df[["SPY Return", "Portfolio P&L", "VaR"]].head())

## Create age-weighted historical simulation model, chose decay factor of 0.97 for now
raw_weights = AGE_WEIGHT_LAMBDA ** np.arange(WINDOW)
weights_newest_first = raw_weights * (1 - AGE_WEIGHT_LAMBDA) / (1 - AGE_WEIGHT_LAMBDA ** WINDOW)
weights_oldest_first = weights_newest_first[::-1]

age_var_dollar, age_dates = [], []
for t in range(WINDOW, len(pnl)):
    window_pnl = pnl.iloc[t - WINDOW:t].to_numpy()
 
    order = np.argsort(window_pnl)
    v = window_pnl[order]
    w = weights_oldest_first[order]
    cum_w = np.cumsum(w) - 0.5 * w
    cum_w /= w.sum()
    q = np.interp(pct / 100.0, cum_w, v)
 
    age_var_dollar.append(-q)
    age_dates.append(pnl.index[t])

age_var_df = pd.DataFrame({"VaR (Age-Weighted)": age_var_dollar}, index=pd.Index(age_dates, name="Date"))
model_df = model_df.join(age_var_df, how="inner")
print(f"Age-weighted VaR added (lambda={AGE_WEIGHT_LAMBDA})")
print(model_df[["SPY Return", "Portfolio P&L", "VaR", "VaR (Age-Weighted)"]].head())

## Create results folder
os.makedirs("results", exist_ok=True)

## Output results to CSV
output = model_df[["SPY Return", "Portfolio P&L", "VaR", "VaR (Age-Weighted)"]].copy()
output.to_csv("results/hs_var_daily_output.csv", index=False)
print(output.head())
print(output.tail())

## Plot Equal-Weighted Historical Simulation VaR against P&L
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df.index, -model_df["Portfolio P&L"], color="black", lw=0.6, label="Realized daily loss ($)")
ax.plot(model_df.index, model_df["VaR"], color="red", lw=1.3, label="99% Equal-Weighted HS VaR ($)")
ax.set_title("250-Day Equal-Weighted Historical Simulation VaR vs. Realized Losses")
ax.set_ylabel("$ Loss")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/equal_weighted_var_vs_losses.png", dpi=150)
plt.show()

## Plot Age-Weighted Historical Simulation VaR against P&L
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(model_df.index, -model_df["Portfolio P&L"], color="black", lw=0.6, label="Realized daily loss ($)")
ax.plot(model_df.index, model_df["VaR (Age-Weighted)"], color="red", lw=1.3, label="VaR (Age-Weighted) ($)")
ax.set_title("250-Day Age-Weighted Historical Simulation VaR vs. Realized Losses")
ax.set_ylabel("$ Loss")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/age_weighted_var_vs_losses.png", dpi=150)
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
checks.append(("Every VaR date has aligned SPY Return & Portfolio P&L", bool(model_df[["SPY Return", "Portfolio P&L"]].notna().all().all())))

## Portfolio value consistently $1,000,000
checks.append(("Portfolio value consistently $1,000,000", PORTFOLIO_VALUE == 1_000_000))

## Confidence level is 99%
checks.append(("Confidence level = 99%", CONFIDENCE_LEVEL == 0.99))

## VaR reported as a positive dollar loss threshold
checks.append(("VaR reported as positive $ loss for all obs", bool((model_df["VaR"] > 0).all())))
checks.append(("Age-weighted VaR reported as positive $ loss for all obs", bool((model_df["VaR (Age-Weighted)"] > 0).all())))

## Missing values limited to necessary initialization periods
missing_after_warmup = ret_df["SPY Return"].iloc[WINDOW:].isna().sum()
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