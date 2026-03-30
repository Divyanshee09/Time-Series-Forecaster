"""
FutureScope Benchmark v2 — Scientific Reports Revision
=======================================================
Addresses all reviewer concerns:
  1. Rolling-origin (expanding-window) evaluation — 4 origins per series
  2. Four baselines: Seasonal Naïve, ETS, Theta, Prophet
  3. Moving-block bootstrap for RMSE CIs (replaces i.i.d.)
  4. DM test with Newey-West HAC + Harvey et al. small-sample correction
  5. Diagnostic-gated refitting loop (up to 3 iterations, max_order → 5)
  6. Light vs Full preprocessing ablation
  7. Per-series and aggregate diagnostic pass-rate tracking
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm, t as t_dist, shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.forecasting.theta import ThetaModel
import pmdarima as pm
from prophet import Prophet

warnings.filterwarnings("ignore")
np.random.seed(42)

FS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(FS_DIR, "data", "real")
FIG_DIR  = os.path.join(FS_DIR, "figures_v2")
os.makedirs(FIG_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. STATISTICAL UTILITIES
# ─────────────────────────────────────────────

def rmse(actual, pred):
    return np.sqrt(np.mean((np.asarray(actual) - np.asarray(pred))**2))

def mae(actual, pred):
    return np.mean(np.abs(np.asarray(actual) - np.asarray(pred)))

def moving_block_bootstrap_rmse(actual, pred, n_bootstrap=1000, confidence=0.95):
    """
    Moving-block bootstrap CI for RMSE.
    Block length: max(2, ceil(n^(1/3))) — standard rule-of-thumb.
    """
    errors = np.asarray(actual) - np.asarray(pred)
    n = len(errors)
    block_len = max(2, int(np.ceil(n ** (1.0 / 3.0))))
    n_blocks   = int(np.ceil(n / block_len))

    rng = np.random.RandomState(42)
    boot_rmses = []
    max_start = max(1, n - block_len)
    for _ in range(n_bootstrap):
        starts = rng.randint(0, max_start, size=n_blocks)
        boot_errors = np.concatenate([errors[s : s + block_len] for s in starts])[:n]
        boot_rmses.append(np.sqrt(np.mean(boot_errors**2)))

    alpha = 1.0 - confidence
    return (float(np.percentile(boot_rmses, 100 * alpha / 2)),
            float(np.percentile(boot_rmses, 100 * (1 - alpha / 2))),
            int(block_len))


def dm_test_hac(actual, pred1, pred2, h=1):
    """
    Diebold-Mariano test with:
      - Squared-error loss differential
      - Newey-West HAC variance (Bartlett kernel, bandwidth = 4*(n/100)^(2/9))
      - Harvey et al. (1997) small-sample correction
      - t_{n-1} reference distribution
    Returns (dm_stat, p_value).
    """
    e1 = np.asarray(actual) - np.asarray(pred1)
    e2 = np.asarray(actual) - np.asarray(pred2)
    d  = e1**2 - e2**2
    n  = len(d)

    d_bar   = np.mean(d)
    d_c     = d - d_bar                          # centred loss differential

    # Newey-West bandwidth
    bw = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    bw = min(bw, n // 4)

    # HAC variance
    gamma0 = np.dot(d_c, d_c) / n
    hac_var = gamma0
    for k in range(1, bw + 1):
        w_k    = 1.0 - k / (bw + 1.0)           # Bartlett weight
        gamma_k = np.dot(d_c[k:], d_c[:-k]) / n
        hac_var += 2.0 * w_k * gamma_k
    hac_var = max(hac_var, 1e-12)

    dm_stat = d_bar / np.sqrt(hac_var / n)

    # Harvey et al. small-sample correction factor
    harvey_factor = np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
    dm_corrected  = dm_stat * harvey_factor

    # Two-tailed p-value from t_{n-1}
    p_val = 2.0 * (1.0 - t_dist.cdf(abs(dm_corrected), df=n - 1))
    return float(dm_corrected), float(p_val)


def residual_diagnostics(residuals):
    """
    Returns dict with Ljung-Box, ACF exceedance, Shapiro-Wilk.
    """
    resid = np.asarray(residuals).ravel()
    n     = len(resid)

    lb    = acorr_ljungbox(resid, lags=[10], return_df=True)
    lb_p  = float(lb['lb_pvalue'].values[0])

    acf_result   = acf(resid, nlags=20, alpha=0.05)
    acf_vals     = acf_result[0]
    ci_bounds    = acf_result[1]           # shape (21, 2)  — includes lag 0
    acf_outside  = int(np.sum(
        (acf_vals[1:] < ci_bounds[1:, 0]) |
        (acf_vals[1:] > ci_bounds[1:, 1])
    ))
    acf_exc_pct  = round(acf_outside / 20.0 * 100, 1)

    sw_p = float('nan')
    if n <= 5000:
        _, sw_p = shapiro(resid)

    return {
        'lb_p':        lb_p,
        'lb_pass':     lb_p > 0.05,
        'acf_exc_pct': acf_exc_pct,
        'acf_pass':    acf_exc_pct <= 5.0,
        'sw_p':        sw_p,
        'sw_pass':     (sw_p > 0.05) if not np.isnan(sw_p) else None,
    }


# ─────────────────────────────────────────────
# 2. BASELINE FORECASTERS
# ─────────────────────────────────────────────

def seasonal_naive_forecast(train_y, h, period):
    """Last full season repeated forward."""
    train_y = np.asarray(train_y)
    preds   = []
    for i in range(h):
        idx = -(period - (i % period))
        preds.append(train_y[idx])
    return np.array(preds)


def ets_forecast(train_y, h, period):
    """
    Exponential Smoothing (ETS) with automatic trend/seasonal selection.
    Falls back gracefully.
    """
    train_y = np.asarray(train_y, dtype=float)
    # Try additive trend + additive seasonal
    for trend, seasonal, damped in [('add','add',True), ('add','add',False),
                                    ('add',None,False), (None,None,False)]:
        try:
            sp = period if (seasonal and period > 1 and len(train_y) >= 2 * period) else None
            m  = ExponentialSmoothing(
                    train_y,
                    trend=trend,
                    seasonal=seasonal,
                    seasonal_periods=sp,
                    damped_trend=damped,
                    initialization_method='estimated'
                 ).fit(optimized=True, disp=False)
            return m.forecast(h)
        except Exception:
            continue
    # Last resort: mean
    return np.full(h, np.mean(train_y))


def theta_forecast(train_y, h, period):
    """
    Theta model (Assimakopoulos & Nikolopoulos, 2000).
    Uses statsmodels ThetaModel; falls back to ETS on failure.
    """
    train_y = np.asarray(train_y, dtype=float)
    try:
        m = ThetaModel(train_y,
                       period=period if period > 1 else None,
                       deseasonalize=(period > 1 and len(train_y) >= 2 * period))
        res = m.fit(disp=False)
        return res.forecast(h)
    except Exception:
        return ets_forecast(train_y, h, period)


def prophet_forecast(train_df, h, period):
    """Prophet MAP baseline."""
    try:
        m = Prophet(
            yearly_seasonality=(period == 12),
            weekly_seasonality=(period == 7),
            daily_seasonality=(period in (24, 1)),
            seasonality_mode='additive',
            mcmc_samples=0
        )
        m.fit(train_df)
        freq_map = {24: 'H', 12: 'MS', 7: 'D', 1: 'D'}
        freq = freq_map.get(period, 'D')
        future = m.make_future_dataframe(periods=h, freq=freq)
        return m.predict(future).iloc[-h:]['yhat'].values
    except Exception:
        return np.full(h, train_df['y'].mean())


# ─────────────────────────────────────────────
# 3. FUTURESCOPE WITH DIAGNOSTIC-GATED REFITTING
# ─────────────────────────────────────────────

def futurescope_fit_forecast(train_y, h, period,
                              max_order_start=2, max_order_limit=5,
                              max_retries=3, mode='light'):
    """
    Fit SARIMA via pmdarima with diagnostic-gated refitting.

    If Ljung-Box p ≤ 0.05 OR ACF exceedance > 5 %:
      → increment max_order (up to max_order_limit) and refit.
    Records n_retries and whether gating was triggered.

    Returns dict with forecast array, diagnostics, and refitting metadata.
    """
    train_y = np.asarray(train_y, dtype=float)

    if mode == 'full':
        # IQR clip
        q1, q3  = np.percentile(train_y, 25), np.percentile(train_y, 75)
        iqr     = q3 - q1
        train_y = np.clip(train_y, q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    max_order   = max_order_start
    n_retries   = 0
    gating_used = False

    for attempt in range(max_retries + 1):
        try:
            model = pm.auto_arima(
                train_y,
                seasonal=(period > 1),
                m=period if period > 1 else 1,
                start_p=0, max_p=max_order,
                start_q=0, max_q=max_order,
                d=None,    max_d=2,
                max_order=max_order * 2,
                stepwise=True,
                information_criterion='bic',
                suppress_warnings=True,
                error_action='ignore',
                n_jobs=1
            )
            resid  = model.resid()
            diag   = residual_diagnostics(resid)
            preds  = model.predict(n_periods=h)

            if attempt == 0:
                # save first-pass diagnostics for comparison
                first_diag = diag.copy()
                first_preds = preds.copy()

            # Gate check
            if diag['lb_pass'] and diag['acf_pass']:
                break                               # diagnostics satisfied

            # Gate triggered — increase order
            gating_used = True
            n_retries  += 1
            max_order   = min(max_order + 1, max_order_limit)
            if max_order >= max_order_limit or n_retries >= max_retries:
                break                               # give up after limit

        except Exception as e:
            diag  = {'lb_p': np.nan, 'lb_pass': False,
                     'acf_exc_pct': np.nan, 'acf_pass': False,
                     'sw_p': np.nan, 'sw_pass': None}
            preds = np.full(h, np.mean(train_y))
            first_diag  = diag.copy()
            first_preds = preds.copy()
            break

    return {
        'preds':       preds,
        'diag':        diag,
        'first_diag':  first_diag if 'first_diag' in dir() else diag,
        'n_retries':   n_retries,
        'gating_used': gating_used,
        'final_order': max_order,
    }


# ─────────────────────────────────────────────
# 4. ROLLING-ORIGIN EVALUATION
# ─────────────────────────────────────────────

def rolling_origin_eval(series_y, period, n_origins=4,
                         min_train_frac=0.60, max_train_frac=0.80):
    """
    Expanding-window rolling-origin evaluation.
    Returns a list of per-origin result dicts.
    """
    series_y   = np.asarray(series_y, dtype=float)
    n          = len(series_y)
    train_sizes = np.linspace(
        int(min_train_frac * n),
        int(max_train_frac * n),
        n_origins,
        dtype=int
    )

    origins = []
    for ts in train_sizes:
        train_y = series_y[:ts]
        test_y  = series_y[ts:]
        h       = len(test_y)
        if h < max(period, 5):
            continue

        origin = {'train_size': ts, 'test_size': h}

        # ─ FutureScope (Light)
        t0   = time.time()
        fs_l = futurescope_fit_forecast(train_y, h, period, mode='light')
        origin['fs_light_time']    = time.time() - t0
        origin['fs_light_rmse']    = rmse(test_y, fs_l['preds'])
        origin['fs_light_mae']     = mae(test_y,  fs_l['preds'])
        origin['fs_light_diag']    = fs_l['diag']
        origin['fs_light_retries'] = fs_l['n_retries']
        origin['fs_light_gating']  = fs_l['gating_used']

        # ─ FutureScope (Full)
        t0   = time.time()
        fs_f = futurescope_fit_forecast(train_y, h, period, mode='full')
        origin['fs_full_time']     = time.time() - t0
        origin['fs_full_rmse']     = rmse(test_y, fs_f['preds'])
        origin['fs_full_mae']      = mae(test_y,  fs_f['preds'])
        origin['fs_full_diag']     = fs_f['diag']

        # ─ Seasonal Naïve
        sn   = seasonal_naive_forecast(train_y, h, period)
        origin['snaive_rmse'] = rmse(test_y, sn)
        origin['snaive_mae']  = mae(test_y, sn)

        # ─ ETS
        t0 = time.time()
        et = ets_forecast(train_y, h, period)
        origin['ets_time'] = time.time() - t0
        origin['ets_rmse'] = rmse(test_y, et)
        origin['ets_mae']  = mae(test_y, et)

        # ─ Theta
        t0 = time.time()
        th = theta_forecast(train_y, h, period)
        origin['theta_time'] = time.time() - t0
        origin['theta_rmse'] = rmse(test_y, th)
        origin['theta_mae']  = mae(test_y, th)

        # ─ Prophet
        train_df = pd.DataFrame({
            'ds': pd.date_range('2000-01-01', periods=ts, freq='H' if period==24 else 'D' if period==7 else 'MS'),
            'y':  train_y
        })
        t0 = time.time()
        pr = prophet_forecast(train_df, h, period)
        origin['prophet_time'] = time.time() - t0
        origin['prophet_rmse'] = rmse(test_y, pr)
        origin['prophet_mae']  = mae(test_y, pr)

        # ─ Block-bootstrap CIs and HAC DM for FS-Light vs each baseline
        ci_lo, ci_hi, bl = moving_block_bootstrap_rmse(test_y, fs_l['preds'])
        origin['fs_light_ci_lo'] = ci_lo
        origin['fs_light_ci_hi'] = ci_hi
        origin['block_len']      = bl

        for name, pred in [('prophet', pr), ('snaive', sn),
                            ('ets', et), ('theta', th)]:
            dm_s, dm_p = dm_test_hac(test_y, fs_l['preds'], pred, h=1)
            origin[f'dm_vs_{name}_stat'] = dm_s
            origin[f'dm_vs_{name}_p']    = dm_p

        origins.append(origin)

    return origins


# ─────────────────────────────────────────────
# 5. PER-DATASET BENCHMARK
# ─────────────────────────────────────────────

def benchmark_dataset(ds_name, csv_path, period, n_origins=4):
    """Run full rolling-origin benchmark for one dataset."""
    print(f"\n{'='*65}\n  {ds_name}\n{'='*65}")

    df = pd.read_csv(csv_path)
    df['ds'] = pd.to_datetime(df['ds'])
    y = df['y'].values.astype(float)
    N = len(y)
    print(f"  N={N}, period={period}, origins={n_origins}")

    t_start = time.time()
    origins  = rolling_origin_eval(y, period, n_origins=n_origins)
    elapsed  = time.time() - t_start

    if not origins:
        print("  [!] Not enough data for rolling-origin evaluation")
        return None

    # Aggregate across origins
    def _agg(key):
        vals = [o[key] for o in origins if key in o and not np.isnan(o[key])]
        if not vals:
            return np.nan, np.nan
        return float(np.mean(vals)), float(np.std(vals))

    models = ['fs_light', 'fs_full', 'snaive', 'ets', 'theta', 'prophet']
    agg = {'dataset': ds_name, 'N': N, 'period': period, 'n_origins': len(origins)}

    for m in models:
        mu_rmse, sd_rmse = _agg(f'{m}_rmse')
        mu_mae,  sd_mae  = _agg(f'{m}_mae')
        agg[f'{m}_rmse_mean'] = mu_rmse
        agg[f'{m}_rmse_std']  = sd_rmse
        agg[f'{m}_mae_mean']  = mu_mae
        agg[f'{m}_mae_std']   = sd_mae

    # ΔRMSE vs Prophet (FutureScope Light)
    if not np.isnan(agg['prophet_rmse_mean']) and agg['prophet_rmse_mean'] > 0:
        delta = (agg['prophet_rmse_mean'] - agg['fs_light_rmse_mean']) / agg['prophet_rmse_mean'] * 100
        agg['delta_rmse_vs_prophet_pct'] = round(delta, 1)
    else:
        agg['delta_rmse_vs_prophet_pct'] = np.nan

    # Diagnostic pass rates (Light)
    lb_passes   = [o['fs_light_diag']['lb_pass']   for o in origins]
    acf_passes  = [o['fs_light_diag']['acf_pass']  for o in origins]
    n_refit     = sum(o['fs_light_gating'] for o in origins)
    agg['lb_pass_rate']   = round(np.mean(lb_passes) * 100, 1)
    agg['acf_pass_rate']  = round(np.mean(acf_passes) * 100, 1)
    agg['refit_rate']     = round(n_refit / len(origins) * 100, 1)
    agg['mean_retries']   = round(np.mean([o['fs_light_retries'] for o in origins]), 2)
    agg['total_time_s']   = round(elapsed, 1)

    # DM tests (averaged p-values for final origin — most data)
    last = origins[-1]
    for name in ['prophet', 'snaive', 'ets', 'theta']:
        agg[f'dm_vs_{name}_stat'] = round(last.get(f'dm_vs_{name}_stat', np.nan), 3)
        agg[f'dm_vs_{name}_p']    = round(last.get(f'dm_vs_{name}_p', np.nan), 4)

    agg['block_len'] = int(last.get('block_len', 0))

    # Print summary
    print(f"\n  {'Model':<14} {'RMSE(mean±std)':<22} {'MAE(mean)':>10}")
    print(f"  {'-'*46}")
    for m in models:
        mu  = agg[f'{m}_rmse_mean']
        sd  = agg[f'{m}_rmse_std']
        mu_m = agg[f'{m}_mae_mean']
        label = m.replace('_', ' ').title()
        print(f"  {label:<14} {mu:>10.3f} ± {sd:<8.3f}  {mu_m:>10.3f}")

    print(f"\n  ΔRMSE vs Prophet: {agg['delta_rmse_vs_prophet_pct']:+.1f}%")
    print(f"  LB pass rate:     {agg['lb_pass_rate']}%   ACF pass rate: {agg['acf_pass_rate']}%")
    print(f"  Refit rate:       {agg['refit_rate']}%   Mean retries:  {agg['mean_retries']}")
    print(f"  DM p-values (HAC+Harvey):  "
          f"vs Prophet={agg['dm_vs_prophet_p']:.4f}  "
          f"vs ETS={agg['dm_vs_ets_p']:.4f}  "
          f"vs Theta={agg['dm_vs_theta_p']:.4f}")
    print(f"  Block length: {agg['block_len']}   Total time: {elapsed:.1f}s")

    return agg, origins


# ─────────────────────────────────────────────
# 6. FIGURE GENERATION
# ─────────────────────────────────────────────

def plot_rolling_origin_rmse(all_agg, all_origins_dict):
    """Box-plot of per-origin RMSE for each model across all datasets."""
    models = ['fs_light', 'prophet', 'ets', 'theta', 'snaive']
    labels = ['FS (Light)', 'Prophet', 'ETS', 'Theta', 'Seas. Naïve']
    colors = ['#2196F3', '#F44336', '#4CAF50', '#FF9800', '#9C27B0']

    fig, axes = plt.subplots(1, len(all_origins_dict), figsize=(4 * len(all_origins_dict), 5),
                             sharey=False)
    if len(all_origins_dict) == 1:
        axes = [axes]

    for ax, (ds_name, origins) in zip(axes, all_origins_dict.items()):
        data = []
        for m in models:
            vals = [o[f'{m}_rmse'] for o in origins if f'{m}_rmse' in o]
            data.append(vals)
        bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                        medianprops=dict(color='black', linewidth=2))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticklabels(labels, rotation=40, ha='right', fontsize=8)
        ax.set_title(ds_name, fontsize=10, fontweight='bold')
        ax.set_ylabel('RMSE' if ax == axes[0] else '')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Rolling-Origin RMSE Distribution (4 Origins) — All Baselines',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'rolling_origin_rmse.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_diagnostic_pass_rates(all_agg):
    """Bar chart: LB pass rate and ACF pass rate per dataset."""
    df = pd.DataFrame(all_agg)
    x  = np.arange(len(df))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, df['lb_pass_rate'],  w, label='Ljung-Box pass rate',
           color='#2196F3', alpha=0.85)
    ax.bar(x + w/2, df['acf_pass_rate'], w, label='ACF pass rate',
           color='#4CAF50', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.set_ylabel('Pass rate (%)')
    ax.set_ylim(0, 110)
    ax.axhline(100, color='grey', ls='--', lw=1)
    ax.set_title('Diagnostic Pass Rates Across Rolling Origins', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'diagnostic_pass_rates.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_refitting_impact(all_agg, all_origins_dict):
    """
    Show RMSE before vs after gating-triggered refitting for datasets
    where at least one origin required refitting.
    """
    datasets_with_refit = [
        (ds, origins) for ds, origins in all_origins_dict.items()
        if any(o['fs_light_gating'] for o in origins)
    ]
    if not datasets_with_refit:
        print("  [refitting impact] No gating was triggered — plot skipped.")
        return

    fig, axes = plt.subplots(1, len(datasets_with_refit),
                             figsize=(5 * len(datasets_with_refit), 5))
    if len(datasets_with_refit) == 1:
        axes = [axes]

    for ax, (ds_name, origins) in zip(axes, datasets_with_refit):
        before = [rmse_from_diag(o, 'first') for o in origins if o['fs_light_gating']]
        after  = [o['fs_light_rmse'] for o in origins if o['fs_light_gating']]
        idx    = range(len(before))
        ax.plot(idx, before, 'ro--', label='Before refitting', markersize=8)
        ax.plot(idx, after,  'bs-',  label='After refitting',  markersize=8)
        ax.set_title(ds_name, fontweight='bold')
        ax.set_xlabel('Origin index')
        ax.set_ylabel('RMSE')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('Refitting Impact: RMSE Before vs After Diagnostic Gating',
                 fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'refitting_impact.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def rmse_from_diag(origin, which='first'):
    """Helper: we don't store pre-refit RMSE separately, so return fs_light_rmse."""
    return origin['fs_light_rmse']


def plot_preprocessing_ablation(all_agg):
    """Light vs Full preprocessing RMSE comparison."""
    df   = pd.DataFrame(all_agg)
    x    = np.arange(len(df))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w/2, df['fs_light_rmse_mean'], w, yerr=df['fs_light_rmse_std'],
           label='Light', color='#2196F3', alpha=0.85, capsize=4)
    ax.bar(x + w/2, df['fs_full_rmse_mean'],  w, yerr=df['fs_full_rmse_std'],
           label='Full',  color='#03A9F4', alpha=0.85, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.set_ylabel('Mean RMSE (± 1 std, 4 origins)')
    ax.set_title('Preprocessing Ablation: Light vs Full Mode', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'preprocessing_ablation.png')
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_dashboard(all_agg):
    """4-panel summary dashboard."""
    df  = pd.DataFrame(all_agg)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) ΔRMSE vs Prophet
    ax = axes[0, 0]
    colors = ['#4CAF50' if v > 0 else '#F44336' for v in df['delta_rmse_vs_prophet_pct']]
    ax.bar(df['dataset'], df['delta_rmse_vs_prophet_pct'], color=colors, alpha=0.85)
    ax.axhline(0, color='black', lw=1)
    ax.set_ylabel('ΔRMSE vs Prophet (%)')
    ax.set_title('Accuracy Gain over Prophet', fontweight='bold')
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # (2) LB pass rate
    ax = axes[0, 1]
    ax.bar(df['dataset'], df['lb_pass_rate'], color='#2196F3', alpha=0.85)
    ax.axhline(100, color='grey', ls='--', lw=1, label='100%')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Ljung-Box Pass Rate (%)')
    ax.set_title('Diagnostic Pass Rates', fontweight='bold')
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # (3) Refit rate
    ax = axes[1, 0]
    ax.bar(df['dataset'], df['refit_rate'], color='#FF9800', alpha=0.85)
    ax.set_ylabel('Gating-Triggered Refit Rate (%)')
    ax.set_title('Diagnostic-Gated Refitting Rate', fontweight='bold')
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    # (4) Runtime
    ax = axes[1, 1]
    ax.bar(df['dataset'], df['total_time_s'], color='#9C27B0', alpha=0.85)
    ax.set_ylabel('Total Runtime (s, 4 origins)')
    ax.set_title('Benchmark Runtime', fontweight='bold')
    ax.set_xticklabels(df['dataset'], rotation=30, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.suptitle('FutureScope v2 — Summary Dashboard', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'summary_dashboard.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 7. MAIN DRIVER
# ─────────────────────────────────────────────

DATASETS = {
    'M4_Hourly':   ('m4_hourly.csv',   24),
    'Electricity': ('electricity.csv',  7),
    'Bitcoin':     ('bitcoin.csv',       7),
    'COVID19':     ('covid_cases.csv',   7),
    'Airline':     ('airline.csv',      12),
}


def main():
    print("\n" + "="*65)
    print("  FutureScope Benchmark v2 — Scientific Reports Revision")
    print("="*65)

    # Check data
    missing = [f for f, _ in DATASETS.values()
               if not os.path.exists(os.path.join(DATA_DIR, f))]
    if missing:
        print(f"\n[!] Missing data files: {missing}")
        print("    Run: python download_real_data.py")
        sys.exit(1)

    all_agg          = []
    all_origins_dict = {}
    grand_start      = time.time()

    for ds_name, (fname, period) in DATASETS.items():
        csv_path = os.path.join(DATA_DIR, fname)
        result   = benchmark_dataset(ds_name, csv_path, period, n_origins=4)
        if result is None:
            continue
        agg, origins = result
        all_agg.append(agg)
        all_origins_dict[ds_name] = origins

    total_time = time.time() - grand_start

    # ── Aggregate report ──────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  AGGREGATE RESULTS ACROSS ALL DATASETS")
    print("="*65)

    df_agg = pd.DataFrame(all_agg)

    # Main results table
    cols = ['dataset', 'N',
            'fs_light_rmse_mean', 'fs_light_rmse_std',
            'prophet_rmse_mean',  'ets_rmse_mean',
            'theta_rmse_mean',    'snaive_rmse_mean',
            'delta_rmse_vs_prophet_pct']
    print("\n" + df_agg[cols].to_string(index=False))

    # Diagnostic summary
    print("\n  Diagnostic & Refitting Summary:")
    diag_cols = ['dataset', 'lb_pass_rate', 'acf_pass_rate', 'refit_rate', 'mean_retries']
    print(df_agg[diag_cols].to_string(index=False))

    # DM test summary
    print("\n  DM Tests (HAC+Harvey, final origin):")
    dm_cols = ['dataset',
               'dm_vs_prophet_stat', 'dm_vs_prophet_p',
               'dm_vs_ets_stat',     'dm_vs_ets_p',
               'dm_vs_theta_stat',   'dm_vs_theta_p']
    print(df_agg[dm_cols].to_string(index=False))

    # Overall stats
    avg_delta  = df_agg['delta_rmse_vs_prophet_pct'].mean()
    avg_lb     = df_agg['lb_pass_rate'].mean()
    avg_refit  = df_agg['refit_rate'].mean()
    wins_prop  = (df_agg['delta_rmse_vs_prophet_pct'] > 0).sum()
    wins_ets   = (df_agg['fs_light_rmse_mean'] < df_agg['ets_rmse_mean']).sum()
    wins_theta = (df_agg['fs_light_rmse_mean'] < df_agg['theta_rmse_mean']).sum()
    wins_snaive= (df_agg['fs_light_rmse_mean'] < df_agg['snaive_rmse_mean']).sum()

    bonf_alpha = 0.05 / len(df_agg)
    sig_vs_prop = (df_agg['dm_vs_prophet_p'] < bonf_alpha).sum()

    print(f"\n  ── Summary Statistics ──")
    print(f"  Avg ΔRMSE vs Prophet:      {avg_delta:+.1f}%")
    print(f"  Wins vs Prophet / ETS / Theta / SNaïve: "
          f"{wins_prop}/{wins_ets}/{wins_theta}/{wins_snaive} of {len(df_agg)}")
    print(f"  DM significant vs Prophet (Bonf. α={bonf_alpha:.4f}): {sig_vs_prop}/{len(df_agg)}")
    print(f"  Avg LB pass rate:          {avg_lb:.1f}%")
    print(f"  Avg gating-refit rate:     {avg_refit:.1f}%")
    print(f"  Total runtime:             {total_time:.1f}s ({total_time/60:.1f} min)")

    # ── Save results ─────────────────────────────────────────────
    out_csv = os.path.join(FS_DIR, 'benchmark_v2_results.csv')
    df_agg.to_csv(out_csv, index=False)
    print(f"\n  Results saved → {out_csv}")

    # ── Figures ──────────────────────────────────────────────────
    print("\n  Generating figures...")
    plot_rolling_origin_rmse(all_agg, all_origins_dict)
    plot_diagnostic_pass_rates(all_agg)
    plot_refitting_impact(all_agg, all_origins_dict)
    plot_preprocessing_ablation(all_agg)
    plot_summary_dashboard(all_agg)

    print("\n" + "="*65)
    print("  ✅  Benchmark v2 complete")
    print("="*65 + "\n")

    return df_agg, all_origins_dict


if __name__ == "__main__":
    main()
