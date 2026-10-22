import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"

# Load model outputs
historical = pd.read_csv(RESULTS / "historical_var_output.csv")
parametric = pd.read_csv(RESULTS / "parametric_var_output.csv")
ewma = pd.read_csv(RESULTS / "ewma_var_output.csv")

# Convert dates
historical["Date"] = pd.to_datetime(historical["Date"])
parametric["Date"] = pd.to_datetime(parametric["Date"])
ewma["Date"] = pd.to_datetime(ewma["Date"])

# Keep only fields needed for comparison
historical = historical[
    ["Date", "SPY_Return", "Portfolio_PnL", "VaR_99"]
].rename(columns={"VaR_99": "Historical_VaR"})

parametric = parametric[
    ["Date", "SPY_Return", "Portfolio_PnL", "VaR_99"]
].rename(columns={"VaR_99": "Parametric_VaR"})

ewma = ewma[
    ["Date", "SPY_Return", "Portfolio_PnL", "VaR_99"]
].rename(columns={"VaR_99": "EWMA_VaR"})

# Merge all three models by Date
combined = (
    historical
    .merge(
        parametric[["Date", "Parametric_VaR"]],
        on="Date",
        how="inner"
    )
    .merge(
        ewma[["Date", "EWMA_VaR"]],
        on="Date",
        how="inner"
    )
)

# Basic checks
assert len(combined) == 3939
assert combined["Date"].is_unique
assert combined["Date"].is_monotonic_increasing
assert combined[
    ["Historical_VaR", "Parametric_VaR", "EWMA_VaR"]
].notna().all().all()

# Save combined dataset
output_file = RESULTS / "combined_var_results.csv"
combined.to_csv(output_file, index=False)

print(combined.head())
print()
print(f"Rows: {len(combined)}")
print(f"Start date: {combined['Date'].min().date()}")
print(f"End date: {combined['Date'].max().date()}")
print(f"Saved to: {output_file}")

# Create figures folder
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# Full-period comparison chart
plt.figure(figsize=(12, 6))

plt.plot(combined["Date"], combined["Historical_VaR"], label="Historical VaR")
plt.plot(combined["Date"], combined["Parametric_VaR"], label="Parametric VaR")
plt.plot(combined["Date"], combined["EWMA_VaR"], label="EWMA VaR")

plt.title("1-Day 99% VaR Model Comparison - $1M SPY Portfolio")
plt.xlabel("Date")
plt.ylabel("VaR ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES / "var_model_comparison_full_period.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# COVID stress-period comparison
stress = combined[
    (combined["Date"] >= "2020-02-01") &
    (combined["Date"] <= "2020-06-30")
]

plt.figure(figsize=(12, 6))

plt.plot(stress["Date"], stress["Historical_VaR"], label="Historical VaR")
plt.plot(stress["Date"], stress["Parametric_VaR"], label="Parametric VaR")
plt.plot(stress["Date"], stress["EWMA_VaR"], label="EWMA VaR")

plt.title("VaR Model Comparison - COVID Stress Period")
plt.xlabel("Date")
plt.ylabel("VaR ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    FIGURES / "var_model_comparison_covid.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()