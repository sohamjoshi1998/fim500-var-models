# fim500-var-models

FIM 500, Fall 2026. We estimate the 1-day 99% Value-at-Risk of a hypothetical $1,000,000 long SPY
portfolio with several VaR methods, then validate each model independently (conceptual soundness,
implementation, benchmarking, backtesting and sensitivity).

## Common specification

| Item | Specification |
|---|---|
| Underlying | SPY, daily |
| Data | Bloomberg `PX_LAST`, Jan 2010 to Aug 31, 2026 (frozen) |
| Returns | Simple, price-only (dividends excluded) |
| Horizon / confidence | 1 day / 99% |
| Initialization | First 250 returns |
| Estimation window | Rolling 250 trading days (Historical, Parametric) |
| Backtesting period | Dec 31, 2010 to Aug 31, 2026 |
| VaR sign convention | Positive dollar loss threshold |

## Models

| Model | Developer | Script |
|---|---|---|
| Historical Simulation | Cassie Wan | in progress on branch `cassie-historical-simulation-var` |
| Parametric (Normal) | Thomas Flaim | `notebooks/03_Parametric_VaR/parametric_var.py` |
| EWMA (λ = 0.94) | Sai Vakkalagadda | `notebooks/04_EWMA_VaR/ewma_var.py` |
| GARCH(1,1) (optional) | unassigned | not yet added |

Project lead: Soham Joshi.

## Repository layout

```
data/
  Var-Spy-daily_raw_data.csv     Bloomberg export, unmodified
  Var-VIX-daily_raw_data.csv     Bloomberg export, unmodified (context only)
notebooks/
  03_Parametric_VaR/             one folder per model
  04_EWMA_VaR/
results/                         model output CSVs and charts (<model>_var_output.csv, *.png)
```

## Data notes

- The SPY file starts in Dec 2004 and is sorted newest first. Rows from 12/10/2004 to 2/16/2005 contain
  VIX values by mistake, and the 2010 start-date filter drops them.
- `Last Price` is the unadjusted close, so returns exclude dividends.
- VIX is not a model input. The EWMA script uses it for one chart that compares EWMA volatility with
  implied volatility and marks the stress periods.

## How to run

Requires Python 3 with pandas, numpy and matplotlib. The parametric model also needs scipy.

```
pip install pandas numpy matplotlib scipy
python notebooks/03_Parametric_VaR/parametric_var.py
python notebooks/04_EWMA_VaR/ewma_var.py
```

Run both from the repository root. The parametric script uses paths relative to the working directory;
the EWMA script resolves its paths from its own location. Each script writes its output CSV and charts
to `results/`.

## EWMA model notes

The variance forecast for day t uses returns through day t-1 only:

```
sigma2[t] = 0.94 * sigma2[t-1] + 0.06 * r[t-1]**2
VaR[t]    = $1,000,000 * 2.326 * sigma[t]
```

The recursion starts from the mean squared return of the first 250 returns, so the first forecast falls
on Dec 31, 2010, the same date as the parametric model. The conditional mean is set to zero. A day is an
exception when the realized loss exceeds that day's VaR.

Output columns in `results/ewma_var_output.csv`: `Date`, `Last Price`, `SPY_Return`, `Portfolio_PnL`,
`EWMA_Volatility`, `VaR_99`, `Loss`, `Exception`.

Known limitations (preliminary):

- Normal quantiles can understate 99% losses because SPY returns have fat tails.
- λ = 0.94 is the RiskMetrics convention for daily data and isn't calibrated to SPY.
- Up and down moves of the same size raise variance equally, so there is no leverage effect.
- Volatility adjusts only after a shock has been observed, and there is no long-run variance level to
  revert to (GARCH has one).
