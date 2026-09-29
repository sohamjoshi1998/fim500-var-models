from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")   # write figures to disk, no window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_SPY = ROOT / "data" / "spy_daily_raw.csv"
RAW_VIX = ROOT / "data" / "vix_daily_raw.csv"   # context only, not a model input
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
OUTPUT = RESULTS / "ewma_var_output.csv"

PORTFOLIO_VALUE = 1_000_000
CONFIDENCE = 0.99
LAMBDA = 0.94
INIT_WINDOW = 250          # returns used only to initialize the variance
START_DATE = "2010-01-01"

Z = NormalDist().inv_cdf(CONFIDENCE)   # one-sided 99% Normal quantile


# 1. Data preparation

def load_spy() -> pd.DataFrame:
    # The raw Bloomberg export starts in Dec 2004 and is sorted newest first. Rows from 12/10/2004 to
    # 2/16/2005 hold VIX values by mistake; the 2010 start-date filter drops them.
    # `Last Price` is the unadjusted close, so returns exclude dividends. Ex-dividend days show a small
    # artificial drop of about 0.3% to 0.4%, which is minor next to a 99% tail move.
    raw = pd.read_csv(RAW_SPY, encoding="utf-8-sig")
    raw["Date"] = pd.to_datetime(raw["Date"], format="%m/%d/%Y")

    spy = (
        raw[["Date", "Last Price"]]
        .sort_values("Date")
        .loc[lambda d: d["Date"] >= START_DATE]
        .reset_index(drop=True)
    )

    assert spy["Date"].is_unique, "duplicate dates"
    assert spy["Last Price"].notna().all() and (spy["Last Price"] > 0).all(), "missing or non-positive prices"
    assert spy["Date"].is_monotonic_increasing

    spy["SPY_Return"] = spy["Last Price"].pct_change()
    spy["Portfolio_PnL"] = PORTFOLIO_VALUE * spy["SPY_Return"]
    return spy.dropna().reset_index(drop=True)   # first price has no return


# 2. Model

def ewma_variance(returns: np.ndarray, lam: float = LAMBDA, init_window: int = INIT_WINDOW) -> np.ndarray:
    # Element t is the variance forecast for day t, using returns[0 .. t-1] only.
    # Entries before init_window are NaN (initialization period).
    sigma2 = np.full(len(returns), np.nan)
    sigma2[init_window] = np.mean(returns[:init_window] ** 2)
    for t in range(init_window + 1, len(returns)):
        sigma2[t] = lam * sigma2[t - 1] + (1 - lam) * returns[t - 1] ** 2
    return sigma2


def build_results(spy: pd.DataFrame, sigma2: np.ndarray) -> pd.DataFrame:
    model = spy[["Date", "Last Price", "SPY_Return", "Portfolio_PnL"]].copy()
    model["EWMA_Volatility"] = np.sqrt(sigma2)
    model["VaR_99"] = PORTFOLIO_VALUE * Z * model["EWMA_Volatility"]
    model["Loss"] = -model["Portfolio_PnL"]
    model["Exception"] = model["Loss"] > model["VaR_99"]
    return model.iloc[INIT_WINDOW:].reset_index(drop=True)   # backtesting period only


# 3. Implementation checks
# Development checks only. Formal backtesting (Kupiec, Christoffersen) belongs to validation.

def run_checks(r: np.ndarray, sigma2: np.ndarray, results: pd.DataFrame) -> None:
    # Hand check against the worked example in the presentation (sigma 1%, return -3%).
    s2_next = LAMBDA * 0.01**2 + (1 - LAMBDA) * (-0.03)**2
    assert np.isclose(s2_next, 0.000148)
    # The slide uses z = 2.326; the exact z = 2.3263 gives slightly higher dollar values.
    assert abs(PORTFOLIO_VALUE * 2.326 * 0.01 - 23_260) < 1
    assert abs(PORTFOLIO_VALUE * 2.326 * np.sqrt(s2_next) - 28_297) < 1

    # No look-ahead: scrambling every return from day t onward must leave forecasts up to day t unchanged.
    t = 2000
    scrambled = r.copy()
    scrambled[t:] = np.random.default_rng(0).permutation(scrambled[t:]) * 5
    assert np.allclose(ewma_variance(scrambled)[: t + 1], sigma2[: t + 1], equal_nan=True)

    # Initialization uses training data only, and the first forecast is on return #251,
    # the same date as the parametric model's first forecast.
    assert np.isclose(sigma2[INIT_WINDOW], np.mean(r[:INIT_WINDOW] ** 2))
    assert np.isnan(sigma2[:INIT_WINDOW]).all()
    assert results["Date"].iloc[0] == pd.Timestamp("2010-12-31")

    # Output has no missing values, VaR is positive, and scaling is $1M x z x sigma.
    assert results[["SPY_Return", "EWMA_Volatility", "VaR_99"]].notna().all().all()
    assert (results["VaR_99"] > 0).all()
    assert np.allclose(results["VaR_99"] / (Z * results["EWMA_Volatility"]), PORTFOLIO_VALUE)

    # Exception flag matches its definition.
    assert (results["Exception"] == (results["Portfolio_PnL"] < -results["VaR_99"])).all()


# 4. Model output

def summarize(results: pd.DataFrame) -> pd.Series:
    n_exc = int(results["Exception"].sum())
    return pd.Series({
        "Backtest days": len(results),
        "Exceptions": n_exc,
        "Expected at 99%": round(len(results) * (1 - CONFIDENCE), 1),
        "Observed rate": f"{n_exc / len(results):.2%}",
        "Mean VaR ($)": f"{results['VaR_99'].mean():,.0f}",
        "Min VaR ($)": f"{results['VaR_99'].min():,.0f}",
        "Max VaR ($)": f"{results['VaR_99'].max():,.0f}",
        "Latest VaR ($)": f"{results['VaR_99'].iloc[-1]:,.0f}",
    }, name="EWMA (lambda = 0.94)")


def by_year(results: pd.DataFrame) -> pd.DataFrame:
    out = results.groupby(results["Date"].dt.year).agg(
        Days=("Exception", "size"),
        Exceptions=("Exception", "sum"),
        Mean_VaR=("VaR_99", "mean"),
    )
    out["Mean_VaR"] = out["Mean_VaR"].round(0)
    return out


def plot_var(df: pd.DataFrame, title: str, path: Path) -> None:
    exc = df[df["Exception"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(df["Date"], df["Portfolio_PnL"], width=1.0, color="0.65", label="Daily P&L")
    ax.plot(df["Date"], -df["VaR_99"], color="tab:blue", lw=1.2, label="-VaR 99% (EWMA)")
    ax.scatter(exc["Date"], exc["Portfolio_PnL"], color="tab:red", s=18, zorder=3,
               label=f"Exceptions ({len(exc)})")
    ax.axhline(0, color="0.3", lw=0.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v/1000:,.0f}k"))
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_vol_vs_vix(results: pd.DataFrame, path: Path) -> None:
    # Context only; VIX is not a model input. It marks the stress periods and lets us compare EWMA's
    # backward-looking volatility with the market's forward-looking implied volatility.
    vix = pd.read_csv(RAW_VIX, encoding="utf-8-sig", usecols=["Date", "Last Price"])
    vix["Date"] = pd.to_datetime(vix["Date"], format="%m/%d/%Y")
    vix = vix.rename(columns={"Last Price": "VIX"})

    ctx = results[["Date", "EWMA_Volatility"]].merge(vix, on="Date", how="left")
    ctx["EWMA_Annualized_%"] = ctx["EWMA_Volatility"] * np.sqrt(252) * 100

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(ctx["Date"], ctx["VIX"], color="0.6", lw=0.8, label="VIX (implied, %)")
    ax.plot(ctx["Date"], ctx["EWMA_Annualized_%"], color="tab:blue", lw=1.0, label="EWMA vol, annualized (%)")
    ax.set_title("EWMA volatility vs VIX")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# Observed behavior (describes the output, no conclusions yet about model adequacy):
# - VaR follows recent volatility closely. It jumps within days of a shock such as March 2020 and falls back
#   over a few weeks once markets calm down, as the ~11-day half-life of lambda = 0.94 implies.
# - VaR moves only after large returns show up, so the first days of a stress episode tend to produce
#   exceptions, and exceptions may cluster.

def main() -> None:
    print(f"z(99%) = {Z:.4f}")

    spy = load_spy()
    print(f"{len(spy):,} daily returns: {spy['Date'].min():%Y-%m-%d} to {spy['Date'].max():%Y-%m-%d}")

    r = spy["SPY_Return"].to_numpy()
    sigma2 = ewma_variance(r)
    results = build_results(spy, sigma2)

    FIGURES.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(f"First forecast date: {results['Date'].iloc[0]:%Y-%m-%d}   rows: {len(results):,}")
    print(f"Saved to: {OUTPUT.relative_to(ROOT).as_posix()}")

    run_checks(r, sigma2, results)
    print("All implementation checks passed.\n")

    print(summarize(results).to_string(), "\n")
    print(by_year(results).to_string(), "\n")

    plot_var(results, "EWMA 1-day 99% VaR vs daily P&L, $1M SPY", FIGURES / "ewma_var_vs_pnl.png")
    covid = results[(results["Date"] >= "2020-01-01") & (results["Date"] <= "2020-06-30")]
    plot_var(covid, "COVID stress period (Jan to Jun 2020)", FIGURES / "ewma_var_covid.png")
    plot_vol_vs_vix(results, FIGURES / "ewma_vol_vs_vix.png")
    print("Figures saved to: results/figures/ewma_*.png")


if __name__ == "__main__":
    main()
