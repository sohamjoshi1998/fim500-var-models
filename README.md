# fim500-var-models

FIM 500, Fall 2026.

## Project objective

We estimate the 1-day 99% Value-at-Risk (VaR) of a long SPY portfolio with three methods: Historical
Simulation, Parametric (Normal) and EWMA. Each model is then validated independently for conceptual
soundness, implementation, benchmarking, backtesting and sensitivity. The goal at this stage is to build
comparable models on the same data and specification. We have not yet concluded which model is best.

## Portfolio specification

| Item | Specification |
|---|---|
| Portfolio | $1,000,000 long SPY, held at a constant $1M exposure |
| Horizon | 1 trading day |
| Confidence level | 99% |
| Daily P&L | $1,000,000 x daily SPY return |
| Returns | Simple, price-only (dividends excluded) |
| Initialization | First 250 returns |
| Backtesting period | Dec 31, 2010 to Aug 31, 2026 |
| VaR sign convention | Positive dollar loss threshold |

## Models

| # | Model | Method | Script |
|---|---|---|---|
| 01 | Historical Simulation | 1st percentile of the last 250 days of P&L. The script also computes an age-weighted version (lambda = 0.97). | `notebooks/01_historical_var/historical_var.py` |
| 02 | Parametric (Normal) | Rolling 250-day mean and standard deviation of returns, VaR = $1M x (2.326 x sigma - mu) | `notebooks/02_parametric_var/parametric_var.py` |
| 03 | EWMA | RiskMetrics variance with lambda = 0.94 and a zero mean, VaR = $1M x 2.326 x sigma | `notebooks/03_ewma_var/ewma_var.py` |

Each forecast uses only data available before that day. All three models make their first forecast on
Dec 31, 2010, after the 250-return initialization period.

The Historical Simulation script currently uses log returns (`RETURN_TYPE = "log"`), while the other two
use simple returns as the specification requires.

## Dataset and source

| File | Contents |
|---|---|
| `data/spy_daily_raw.csv` | SPY daily prices from Bloomberg (`PX_LAST` as `Last Price`, plus high, low and open). Unmodified export. |
| `data/vix_daily_raw.csv` | VIX daily levels from Bloomberg. Used for context only, not as a model input. |

The dataset is frozen at Aug 31, 2026 and every model filters it to Jan 2010 onward.

- The SPY export starts in Dec 2004 and is sorted newest first. Rows from 12/10/2004 to 2/16/2005 contain
  VIX values by mistake, and the 2010 start-date filter drops them.
- `Last Price` is the unadjusted close, so returns exclude dividends. Ex-dividend days show a small
  artificial drop of about 0.3% to 0.4%.

## Folder structure

Folders and files use lowercase snake_case. Model folders are numbered in the order above, and every output
starts with the model name.

```
data/
  spy_daily_raw.csv
  vix_daily_raw.csv
notebooks/
  01_historical_var/historical_var.py
  02_parametric_var/parametric_var.py
  03_ewma_var/ewma_var.py
results/
  historical_var_output.csv
  parametric_var_output.csv
  ewma_var_output.csv
  figures/
    historical_var_equal_weighted.png
    historical_var_age_weighted.png
    parametric_var_over_time.png
    parametric_var_vs_pnl.png
    ewma_var_vs_pnl.png
    ewma_var_covid.png
    ewma_vol_vs_vix.png
requirements.txt
README.md
```

New work should follow the same pattern: `notebooks/NN_<model>_var/<model>_var.py`, output in
`results/<model>_var_output.csv`, and charts in `results/figures/<model>_var_<description>.png`.

## How to run

You need Python 3. Install the packages once:

```
pip install -r requirements.txt
```

Run each model from the repository root, because the scripts read from `data/` and write to `results/`
using paths relative to the root:

```
python notebooks/01_historical_var/historical_var.py
python notebooks/02_parametric_var/parametric_var.py
python notebooks/03_ewma_var/ewma_var.py
```

Each script prints its implementation checks and overwrites its own CSV and charts in `results/`. The
Historical and Parametric scripts open each chart in a window, and the script continues once you close it.
The EWMA script saves its charts without displaying them.

## Team and model assignments

| Role | Name | GitHub |
|---|---|---|
| Project lead | Soham Joshi | `sohamjoshi1998` |
| Historical Simulation VaR | Cassie Wan | `cassiemwan` |
| Parametric VaR | Thomas Flaim | `Tflaim23` |
| EWMA VaR, repository structure and README | Sai Vakkalagadda | `subhash-vakkalagadda` |

## Current project workflow

1. Each piece of work lives on its own branch, named after the owner and the task (for example
   `thomas-parametric-var`).
2. When it is ready, the owner opens a pull request into `main`. The project lead reviews and merges it.
3. Scripts and their outputs are committed together, so `results/` always matches the code on `main`.

Model development is done for all three models. The open tasks are:

- Model comparison: combine the three outputs by date into `combined_var_results.csv`, and chart the three
  VaR series over the full backtesting period and over one stress period.
- EDA and presentation figures: SPY price and return trends, the return distribution, rolling volatility
  and the main stress periods, with a few observations on what they mean for the models.
- Model documentation for each model: development summary, methodology, key assumptions, parameters,
  rolling VaR chart and known limitations.

Formal backtesting (Kupiec and Christoffersen tests) and the rest of the independent validation come after
these tasks.
