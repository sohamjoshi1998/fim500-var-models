import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

PORTFOLIO_VALUE = 1_000_000
LOOKBACK_WINDOW = 250
CONFIDENCE_LEVEL = 0.99
Z_SCORE = norm.ppf(CONFIDENCE_LEVEL)

# Load SPY data
spy = pd.read_csv("data/Var-Spy-daily_raw_data.csv")

# Convert Date column to datetime
spy["Date"] = pd.to_datetime(spy["Date"])

# Keep project period only
spy = spy[spy["Date"] >= "2010-01-01"]

# Sort oldest to newest
spy = spy.sort_values("Date").reset_index(drop=True)

# Calculate daily simple returns from Last Price
spy["SPY_Return"] = spy["Last Price"].pct_change()

# Calculate daily profit/loss for the $1 million SPY portfolio
spy["Portfolio_PnL"] = PORTFOLIO_VALUE * spy["SPY_Return"]

# Estimate rolling 250-day mean and standard deviation
# Shift by 1 day so each forecast uses only information available before that date
spy["Rolling_Mean"] = (
    spy["SPY_Return"]
    .rolling(LOOKBACK_WINDOW)
    .mean()
    .shift(1)
)

spy["Rolling_Std"] = (
    spy["SPY_Return"]
    .rolling(LOOKBACK_WINDOW)
    .std()
    .shift(1)
)

# Calculate 1-day 99% Parametric Normal VaR in dollar terms
spy["VaR_99"] = PORTFOLIO_VALUE * (
    Z_SCORE * spy["Rolling_Std"] - spy["Rolling_Mean"]
)

# Basic implementation checks
valid_var = spy["VaR_99"].dropna()

print(
    f"First VaR forecast date: "
    f"{spy.loc[spy['VaR_99'].notna(), 'Date'].iloc[0]}"
)
print(f"Number of VaR forecasts: {len(valid_var)}")
print(f"Minimum VaR: ${valid_var.min():,.2f}")
print(f"Maximum VaR: ${valid_var.max():,.2f}")
print(f"Negative VaR values: {(valid_var < 0).sum()}")

# Create standardized Parametric VaR output
model_output = spy.loc[
    spy["VaR_99"].notna(),
    [
        "Date",
        "Last Price",
        "SPY_Return",
        "Portfolio_PnL",
        "Rolling_Mean",
        "Rolling_Std",
        "VaR_99"
    ]
].copy()

# Create results folder if it does not already exist
os.makedirs("results", exist_ok=True)

# Save output to the results folder
model_output.to_csv(
    "results/parametric_var_output.csv",
    index=False
)

print("\nModel output preview:")
print(model_output.head())

print(f"\nOutput rows saved: {len(model_output)}")
print("Saved to: results/parametric_var_output.csv")

# Plot daily 99% VaR over time
plt.figure(figsize=(12, 6))

plt.plot(
    model_output["Date"],
    model_output["VaR_99"]
)

plt.title("1-Day 99% Parametric VaR for $1M SPY Portfolio")
plt.xlabel("Date")
plt.ylabel("VaR ($)")
plt.grid(True)

plt.tight_layout()

# Save VaR chart
plt.savefig(
    "results/parametric_var_over_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Plot daily portfolio P&L against the 99% VaR threshold
plt.figure(figsize=(12, 6))

plt.plot(
    model_output["Date"],
    model_output["Portfolio_PnL"],
    label="Daily Portfolio P&L",
    linewidth=0.8
)

plt.plot(
    model_output["Date"],
    -model_output["VaR_99"],
    label="99% VaR Threshold",
    linewidth=1.2
)

plt.title("Daily Portfolio P&L vs. 99% Parametric VaR")
plt.xlabel("Date")
plt.ylabel("Dollar P&L ($)")
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save P&L vs VaR chart
plt.savefig(
    "results/parametric_var_vs_pnl.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Basic data quality checks
print(f"Duplicate dates: {spy['Date'].duplicated().sum()}")
print(f"Missing Last Price values: {spy['Last Price'].isna().sum()}")
print(f"Data start date: {spy['Date'].min()}")
print(f"Data end date: {spy['Date'].max()}")
print(f"Raw observations used: {len(spy)}")

# Hard implementation checks
assert spy["Date"].duplicated().sum() == 0, "Duplicate dates found."
assert spy["Last Price"].isna().sum() == 0, "Missing price values found."
assert (valid_var >= 0).all(), "Negative VaR values found."
assert spy.loc[spy["VaR_99"].notna(), "Date"].iloc[0] == pd.Timestamp("2010-12-31")

print("All implementation checks passed.")