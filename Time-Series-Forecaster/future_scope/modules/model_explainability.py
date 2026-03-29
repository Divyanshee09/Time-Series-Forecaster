import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


def residual_diagnostics(model_fit, ts, target_col, seasonal_period=None, max_acf_lags=40):
    """
    Run residual checks (Ljung-Box), plot residual ACF and series ACF, and generate textual insights.

    Returns: (fig_res_acf, fig_series_acf, insights_list, residuals_white_noise_bool)
    """

    insights = []
    resid = None
    fig_res_acf = None
    fig_series_acf = None

    # Try to extract residuals from fitted model
    try:
        if model_fit is not None and hasattr(model_fit, "resid"):
            resid = pd.Series(model_fit.resid).dropna()
    except Exception:
        resid = None

    # If residuals unavailable
    if resid is None or resid.empty:
        insights.append("Could not extract residuals from model.")
        resid = ts[target_col].dropna()

    # Ljung-Box Test
    residuals_white_noise = False
    try:
        lb_lags = min(10, max(1, len(resid) // 5))
        lb_res = acorr_ljungbox(resid, lags=[lb_lags], return_df=True)
        lb_pvalue = float(lb_res["lb_pvalue"].iloc[0])

        if lb_pvalue > 0.05:
            insights.append(
                f"Ljung-Box test (lag={lb_lags}): p={lb_pvalue:.3f} — residuals appear to be white noise."
            )
            residuals_white_noise = True
        else:
            insights.append(
                f"Ljung-Box test (lag={lb_lags}): p={lb_pvalue:.3f} — residuals show significant autocorrelation."
            )

    except Exception as e:
        insights.append(f"Ljung-Box test failed: {e}")

    # Residual ACF Plot
    try:
        max_acf = min(max_acf_lags, len(resid) - 1)
        res_acf_vals = acf(resid, nlags=max_acf, fft=True)

        conf = 1.96 / np.sqrt(len(resid))

        fig_res_acf = go.Figure()

        fig_res_acf.add_trace(
            go.Bar(x=list(range(len(res_acf_vals))), y=res_acf_vals, name="ACF")
        )

        # Confidence bands
        fig_res_acf.add_hline(y=conf, line_dash="dash", line_color="red")
        fig_res_acf.add_hline(y=-conf, line_dash="dash", line_color="red")

        fig_res_acf.update_layout(
            title="Residual ACF",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
        )

    except Exception as e:
        insights.append(f"Failed to compute residual ACF: {e}")

    # Series ACF
    try:
        series = ts[target_col].dropna()
        max_series_lags = min(max_acf_lags, len(series) - 1)

        series_acf_vals = acf(series, nlags=max_series_lags, fft=True)

        conf = 1.96 / np.sqrt(len(series))

        fig_series_acf = go.Figure()

        fig_series_acf.add_trace(
            go.Bar(x=list(range(len(series_acf_vals))), y=series_acf_vals, name="ACF")
        )

        # Confidence bands
        fig_series_acf.add_hline(y=conf, line_dash="dash", line_color="red")
        fig_series_acf.add_hline(y=-conf, line_dash="dash", line_color="red")

        fig_series_acf.update_layout(
            title="Series ACF",
            xaxis_title="Lag",
            yaxis_title="Autocorrelation",
        )

        # Seasonal Strength
        try:
            sp = seasonal_period if seasonal_period is not None else 365
            stl = STL(series, period=sp)
            res = stl.fit()

            seasonal_strength = (
                res.seasonal.var() / series.var() if series.var() != 0 else 0
            )

            if seasonal_strength >= 0.1:
                if sp in (7, 14):
                    label = "weekly"
                elif sp in (30, 31):
                    label = "monthly"
                elif sp >= 360:
                    label = "yearly"
                else:
                    label = f"period={sp}"

                insights.append(f"The series shows strong {label} seasonality.")
            else:
                insights.append("No strong seasonal component detected.")

        except Exception:
            pass

        # Significant Lags
        try:
            significant_lags = []

            for lag, val in enumerate(series_acf_vals):
                if lag == 0:
                    continue
                if abs(val) >= conf:
                    significant_lags.append((lag, val))

            if significant_lags:
                significant_lags = sorted(
                    significant_lags, key=lambda x: -abs(x[1])
                )[:5]

                for lag, val in significant_lags:
                    insights.append(
                        f"Lag-{lag} autocorrelation is high (r={val:.2f}), meaning the current value depends on the value {lag} period(s) ago."
                    )
            else:
                abs_vals = [
                    (lag, val)
                    for lag, val in enumerate(series_acf_vals)
                    if lag > 0
                ]

                if abs_vals:
                    max_lag, max_val = max(abs_vals, key=lambda x: abs(x[1]))

                    insights.append(
                        f"No very high autocorrelations found; strongest is lag-{max_lag} (r={max_val:.2f})."
                    )

        except Exception as e:
            insights.append(f"ACF insight generation failed: {e}")

    except Exception as e:
        insights.append(f"Failed to compute series ACF or generate insights: {e}")

    return fig_res_acf, fig_series_acf, insights, residuals_white_noise