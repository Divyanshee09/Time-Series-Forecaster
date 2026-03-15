# FutureScope: Diagnostic-First Time Series Forecaster - Research Package

> **TMLR Submission**: A transparent forecasting framework achieving competitive accuracy with Prophet while providing full diagnostic transparency

---

## 🎯 Quick Start (30 seconds)

```bash
# Setup
conda create -n timeseries python=3.10 -y
conda activate timeseries
pip install -r requirements.txt

# Run benchmark (8 minutes)
python benchmark_final.py

# Results: benchmark_final_results.csv + figures_final/
```

## 📊 Benchmark Results Summary

| Metric | Value |
|--------|-------|
| **Datasets** | 5 (hourly, daily, monthly) |
| **Avg RMSE vs Prophet** | Within 10% (competitive) |
| **Runtime** | 8.3 minutes (single CPU) |
| **Diagnostic Depth** | Full (Ljung-Box, ACF, residuals) |

**Key Finding**: FutureScope matches Prophet's accuracy on 3/5 datasets (p>0.05) while being **670x slower but 100% transparent**.

---

## 📁 Files Overview

| File | Description | LOC |
|------|-------------|-----|
| `future_scope_fixed.py` | Core forecaster class | 325 |
| `benchmark_final.py` | Optimized benchmark suite | 350 |
| `TMLR_SUBMISSION_REPORT.md` | Full research report | - |
| `demo.ipynb` | Interactive tutorial | - |
| `requirements.txt` | Dependencies | 16 |

---

## 🔬 Key Research Contributions

1. **Methodological**: Classical SARIMA + modern automation = competitive performance
2. **Transparency**: Full residual diagnostics missing from Prophet/XGBoost
3. **Reproducibility**: One-command benchmark (<10 min runtime)

---

## 📈 Performance Comparison

```
RMSE Improvement over Prophet:
  M4 Hourly:    -37% (FS better) ✓
  Electricity:  -22% (FS better) ✓
  Traffic:       +7% (comparable) ~
  Airline:      -15% (comparable) ~
  Synthetic:    +17% (Prophet better) ✗

Average: Competitive (within 10%)
```

---

## 🚀 Usage Example

```python
from future_scope_fixed import FutureScopeForecaster

fs = FutureScopeForecaster(target_col='y', datetime_col='ds', seasonal_period=24)
fs.ingest_data('data.csv')
fs.preprocess(light_mode=True)
fs.select_model(mode='simple', max_order=2)

forecast = fs.forecast(horizon=48)
diagnostics = fs.diagnostics()  # Plotly figure with residual analysis
```

---

## 🎓 Citation

```bibtex
@article{futurescope2026,
  title={FutureScope: Diagnostic-First Time Series Forecasting},
  journal={TMLR},
  year={2026}
}
```

---

See **[TMLR_SUBMISSION_REPORT.md](TMLR_SUBMISSION_REPORT.md)** for full details.
