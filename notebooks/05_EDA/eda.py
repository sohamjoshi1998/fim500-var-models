import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

SPY_PRICES = "data/spy_daily_raw.csv"
VIX_PRICES = "data/vix_daily_raw.csv"
HS_DATA = "results/historical_var_output.csv"
PARAMETRIC_DATA = "results/parametric_var_output.csv"
EWMA_DATA = "results/ewma_var_output.csv"
DATA_START = "2010-01-01"
PORTFOLIO_VALUE = 1_000_000
CONFIDENCE = 0.99

## Loaded in data and model output files
spy = pd.read_csv(SPY_PRICES, encoding="utf-8-sig")
vix = pd.read_csv(VIX_PRICES, encoding="utf-8-sig")
hs = pd.read_csv(HS_DATA, encoding="utf-8-sig")
parametric = pd.read_csv(PARAMETRIC_DATA, encoding="utf-8-sig")
ewma = pd.read_csv(EWMA_DATA, encoding="utf-8-sig")

## Cleaned data, converted date column to datetime, and sorted chronologically
def clean_data(df, price_col):
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df = df.dropna(subset=[price_col]).reset_index(drop=True)
    df = df.loc[df.groupby("Date")[price_col].idxmax()].sort_values("Date").reset_index(drop=True)
    df = df[df["Date"] >= pd.Timestamp(DATA_START)].reset_index(drop=True)
    return df

spy = clean_data(spy, "Last Price")
vix = clean_data(vix, "Last Price")
hs = clean_data(hs, "SPY_Return")
parametric = clean_data(parametric, "SPY_Return")
ewma = clean_data(ewma, "SPY_Return")
print(spy.head())
print(vix.head())
print(hs.head())
print(parametric.head())
print(ewma.head())

## Plot SPY price and returns daily trends
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
ax[0].plot(spy["Date"], spy["Last Price"], lw=1)
ax[0].set_title("SPY price")
ax[0].set_ylabel("Price ($)")
ax[0].margins(x=0)
ax[1].plot(parametric["Date"], parametric["SPY_Return"], lw=0.5, color="red")
ax[1].set_title("SPY daily simple returns")
ax[1].set_ylabel("Return (%)")
ax[1].margins(x=0)
plt.tight_layout()
plt.savefig("results/figures/eda_price_return_trends.png", dpi=150)
plt.show()

## Make sure all model outputs have the same number of observations and dates are aligned
assert len(hs) == len(parametric) == len(ewma)
hs["Date"] = parametric["Date"].values     

## Calculate z-score
z = stats.norm.ppf(CONFIDENCE)

## Create a summary table of model outputs
model_outputs = pd.DataFrame({
    "Date": parametric["Date"], "ret": parametric["SPY_Return"],
    "VaR_HS": hs["VaR_AW"].values,
    "VaR_Para": parametric["VaR_99"].values,
    "VaR_EWMA": ewma["VaR_99"].values,
    "mu_para": parametric["Rolling_Mean"].values,
    "sd_para": parametric["Rolling_Std"].values,
    "sd_ewma": ewma["EWMA_Volatility"].values,
}).set_index("Date")

## Calculate standardardized returns for each model
std_returns = pd.DataFrame({
    "Historical": model_outputs["ret"] / (model_outputs["VaR_HS"] / PORTFOLIO_VALUE / z),
    "Parametric": (model_outputs["ret"] - model_outputs["mu_para"]) / model_outputs["sd_para"],
    "EWMA": model_outputs["ret"] / model_outputs["sd_ewma"],
})

## Create summary table of statistics for each model
loss = -PORTFOLIO_VALUE * model_outputs["ret"]
rows = {}
for m, col in [("Historical", "VaR_HS"), ("Parametric", "VaR_Para"), ("EWMA", "VaR_EWMA")]:
    s, v = std_returns[m], model_outputs[col]
    exc = loss > v
    rows[m] = {
        "Mean VaR ($)": v.mean(), "Min VaR ($)": v.min(), "Max VaR ($)": v.max(),
        "VaR STD ($)": v.std(), "Max/Min VaR": v.max() / v.min(),
        "Exceptions (Loss > VaR)": int(exc.sum()), "Exception rate": exc.mean(),
        "Std of returns": s.std(), "Skew": stats.skew(s),
        "Excess kurtosis": stats.kurtosis(s),
        "1% quantile": s.quantile(0.01),
        "Days < -3": int((s < -3).sum()),
    }
summary = pd.DataFrame(rows)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(summary.to_string())

## Plot of standardized returns for each model
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for j, m in enumerate(std_returns):
    s = std_returns[m].dropna()
    ax[0, j].hist(s, bins=120, density=True, alpha=0.6)
    xs = np.linspace(-6, 6, 400)
    ax[0, j].plot(xs, stats.norm.pdf(xs), "r")

    ## Add vertical lines for empirical 1% quantile and normal 1% quantile
    ax[0, j].axvline(summary.loc["1% quantile", m], color="k", ls="--", label=f"Empirical 1% ({summary.loc["1% quantile", m]:.2f})")
    ax[0, j].axvline(-z, color="r", ls="--", label=f"Normal 1% ({-z:.2f})")
    ax[0, j].set_xlim(-6, 6)
    ax[0, j].set_title(f"{m}: standardized returns vs normal distribution")
    ax[0, j].legend(fontsize=8, loc="upper left")

    ## Add QQ plot to show normality of standardized returns
    stats.probplot(s, dist="norm", plot=ax[1, j])
    ax[1, j].set_title(f"{m}: QQ plot")
plt.tight_layout()
plt.savefig("results/figures/eda_return_distribution.png", dpi=150)
plt.show()

## Calculate volatility for each model (except HS as there is no volatility estimate)
vol_para = parametric.set_index("Date")["Rolling_Std"] * np.sqrt(250)
vol_ewma = ewma.set_index("Date")["EWMA_Volatility"] * np.sqrt(250)

## Calculate 21-day rolling volatility to capture short-term volatility spikes (21 days = 1 month of trading days)
returns = parametric.set_index("Date")["SPY_Return"]
vol_21 = returns.rolling(21).std() * np.sqrt(250)  

## Match VIX dates to the volatility estimates dates
vix = vix.set_index("Date")["Last Price"].sort_index().reindex(vol_ewma.index)

## Plot volatility estimates with VIX
fig, ax = plt.subplots(figsize=(14, 6))
ax.fill_between(vix.index, vix / 100, color="0.85", lw=0, label="VIX / 100 (context)", zorder=0)
ax.plot(vol_21, color="red", lw=0.7, alpha=0.9, label="21-day realized", zorder=1)
ax.plot(vol_para, color="blue", lw=0.7, label="Parametric (250-day window)", zorder=2)
ax.plot(vol_ewma, color="green", lw=0.7, label="EWMA (λ = 0.94)", zorder=3)
ax.set_title("Annualized volatility: 21-day realized vs 250-day window vs EWMA (VIX for context)")
ax.set_ylabel("Annualized volatility")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_ylim(0, None)
ax.grid(alpha=0.25)
ax.legend(loc="upper left", frameon=True)
ax.margins(x=0)
plt.tight_layout()
plt.savefig("results/figures/eda_rolling_volatility.png", dpi=150)
plt.show()

## Flag days where 21-day volatility is above the 95th percentile
threshold = vol_21.quantile(0.95)
stress = vol_21 > threshold

## Group flagged days into periods (flagged days within 21-1 trading days of each other are merged)
price = parametric.set_index("Date")["Last Price"]
period, start, prev = [], None, None
for d in stress[stress].index:
    if start is None:
        start = prev = d
    elif returns.index.get_loc(d) - returns.index.get_loc(prev) <= 20:
        prev = d
    else:
        period.append((start, prev))
        start = prev = d
if start is not None:
    period.append((start, prev))

## Create summary table of stress periods with peak volatility, worst day, and max drawdown of each period
rows = []
for s, e in period:
    seg, px = returns.loc[s:e], price.loc[s:e]
    peak = vol_21.loc[s:e].idxmax()
    rows.append({
        "Start": s.date(), "End": e.date(), "Days": len(seg),
        "Peak date": peak.date(),
        "Peak 21d vol": vol_21.loc[peak],
        "250d vol at peak": vol_para.loc[peak],
        "EWMA vol at peak": vol_ewma.loc[peak],
        "VIX at peak": vix.loc[peak],
        "Worst day": seg.min(),
        "Max drawdown": (px / px.cummax() - 1).min(),
    })
stress_table = pd.DataFrame(rows)
print(f"Threshold (95th percentile of 21d vol): {threshold:.1%}")
print(stress_table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

## Plot 21-day volatility with stress periods shaded
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(vol_21, lw=0.8, label="21-day realized vol")
ax.axhline(threshold, color="r", ls="--", label=f"95th percentile ({threshold:.1%})")
for s, e in period:
    ax.axvspan(s, e, color="red", alpha=0.2)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Stress periods (shaded)")
ax.legend()
ax.margins(x=0)
plt.tight_layout()
plt.savefig("results/figures/eda_stress_periods.png", dpi=150)
plt.show()