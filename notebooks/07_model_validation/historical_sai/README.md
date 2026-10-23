# Historical Simulation validation - Sai Vakkalagadda

FIM 500 | October 23, 2026. Independent conceptual review, implementation checks,
backtesting, benchmarking and sensitivity for Cassie Wan's Historical Simulation model.
No final effectiveness rating is assigned.

## Files to commit

- `Historical_Backtesting_Benchmarking_Sai.ipynb`: executed notebook, calculations and charts.
- `results/provenance.json`: SHA-256 hashes for the reviewed repository inputs.
- `baseline_commit.txt`, `requirements_validation.txt`, this README and `.gitignore`.
- Report at `../../../documentation/validation/Historical_Validation_Report_Sai.pdf`.

Local snapshot, standalone script, exported CSVs/charts, duplicate report, editable report
draft, build helper, meeting notes and runtime files are ignored. They are not required to run the submitted analysis.

## Reproduce

From the repository root:

```powershell
python -m pip install -r notebooks/07_model_validation/historical_sai/requirements_validation.txt
```

Open `Historical_Backtesting_Benchmarking_Sai.ipynb` in a notebook editor such as VS Code,
select this Python environment and Run All. The notebook reads
existing repository data/model outputs and verify their hashes against
`results/provenance.json` before analysis. It regenerates CSVs and figures inside this
package's `results/`; these generated files stay ignored. Saved notebook outputs already
include the analysis tables and charts.
The reviewed baseline is commit `bd49450e23f0a4d64703ba90298ceb4cc8e8ccf5`.
If inputs change, the check intentionally fails: review the new model/data and establish
new provenance deliberately rather than silently replacing the reviewed baseline.

## Findings

Historical: 59 exceptions in 3,939 forecasts (1.4978%) against 39.39 expected.
Kupiec p=0.003448; Christoffersen independence p=0.00002116. Both reject at 5%.
All Historical forecasts independently reproduce within $0.000001. Age-weighted Historical
is supplementary. Sensitivity uses a common shorter calendar for different lookback windows.

The notebook documents five manual date checks, transition counts, annual exception
breakdowns, stress comparisons and preliminary limitations. The PDF explains the findings.
No external downloads are needed to rerun; the repository already contains the inputs.
