import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import warnings

warnings.filterwarnings("ignore")

class FutureScopeForecaster:
    def __init__(self, target_col=None, datetime_col=None, freq=None, seasonal_period=12):
        self.target_col = target_col
        self.datetime_col = datetime_col
        self.freq = freq
        self.seasonal_period = seasonal_period
        self.data = None
        self.original_data = None
        self.best_model = None
        self.model_fit = None
        self.history = {}
        self.transformations = []
        self.last_actual_value = None
        self.first_actual_value = None

    def ingest_data(self, file_path_or_df, datetime_col=None, target_col=None):
        """Standardizes input formats and handles initial loading."""
        if isinstance(file_path_or_df, str):
            if file_path_or_df.endswith('.csv'):
                self.data = pd.read_csv(file_path_or_df)
            elif file_path_or_df.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path_or_df)
            else:
                raise ValueError("Unsupported file format. Please provide CSV or Excel.")
        elif isinstance(file_path_or_df, pd.DataFrame):
            self.data = file_path_or_df.copy()
        
        self.original_data = self.data.copy()
        self.datetime_col = datetime_col or self.datetime_col
        self.target_col = target_col or self.target_col
        
        if self.datetime_col:
            self.data[self.datetime_col] = pd.to_datetime(self.data[self.datetime_col])
            self.data = self.data.sort_values(self.datetime_col)
        
        return self.data

    def preprocess(self, freq=None):
        """Integrated cleaning pipeline: alignment, imputation, and outlier management."""
        if self.data is None:
            raise ValueError("No data ingested. Call ingest_data first.")
        
        df = self.data.copy()
        
        # 1. Temporal Alignment & Gap Filling
        if self.datetime_col:
            df = df.set_index(self.datetime_col)
            if freq:
                self.freq = freq
            else:
                self.freq = pd.infer_freq(df.index) or 'D'
            
            df = df.resample(self.freq).asfreq()
            # Intelligent Imputation
            df[self.target_col] = df[self.target_col].interpolate(method='time')
        
        # 2. Duplicate Resolution (already handled by resample if unique, but for safety)
        df = df[~df.index.duplicated(keep='first')]
        
        # 3. Scalable Outlier Management
        n_samples = len(df)
        series = df[self.target_col]
        
        if n_samples < 10000:
            # Small Dataset Logic
            # Normality check
            _, p_val = stats.shapiro(series.dropna()) if n_samples <= 5000 else (0, 0)
            skewness = stats.skew(series.dropna())
            
            if p_val > 0.05 and abs(skewness) < 0.5:
                # Z-Score
                z_scores = np.abs(stats.zscore(series))
                outlier_mask = z_scores > 3
            else:
                # IQR
                q1, q3 = series.quantile([0.25, 0.75])
                iqr = q3 - q1
                outlier_mask = (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
            
            # Treatment
            if n_samples < 5000:
                df.loc[outlier_mask, self.target_col] = series.median()
            else:
                # Winsorization
                lower = series.quantile(0.05)
                upper = series.quantile(0.95)
                df[self.target_col] = np.clip(series, lower, upper)
        else:
            # Large Dataset: Isolation Forest
            iso = IsolationForest(contamination=0.05, random_state=42)
            preds = iso.fit_predict(series.values.reshape(-1, 1))
            outlier_mask = preds == -1
            df.loc[outlier_mask, self.target_col] = series.median() # Robust replacement

        # 4. Non-Linear Corrections
        skewness = stats.skew(df[self.target_col])
        if abs(skewness) > 1.0:
            pt = PowerTransformer(method='yeo-johnson')
            df[self.target_col] = pt.fit_transform(df[[self.target_col]])
            self.transformations.append(('power_transform', pt))
            
        self.data = df
        return self.data

    def decompose(self, method='stl'):
        """Triple Decomposition Framework."""
        if self.data is None:
            raise ValueError("Data not preprocessed.")
        
        series = self.data[self.target_col]
        if method == 'stl':
            stl = STL(series, period=self.seasonal_period)
            res = stl.fit()
            return res
        else:
            return seasonal_decompose(series, model='additive', period=self.seasonal_period)

    def _check_stationarity(self, series):
        res = adfuller(series.dropna())
        return res[1] < 0.05 # p-value < 0.05

    def select_model(self, p_range=range(0, 3), d_range=range(0, 2), q_range=range(0, 3)):
        """Hybrid Model Selection Architecture."""
        series = self.data[self.target_col]
        
        # Phase A: Seasonal Strength
        stl_res = self.decompose(method='stl')
        seasonal_strength = max(0, 1 - np.var(stl_res.resid) / np.var(stl_res.seasonal + stl_res.resid))
        
        is_seasonal = seasonal_strength >= 0.1
        self.history['seasonal_strength'] = seasonal_strength
        
        # Stationarity Engine
        d = 0
        diff_series = series.copy()
        while not self._check_stationarity(diff_series) and d < 3:
            if d == 0:
                self.first_actual_value = series.iloc[0]
            diff_series = diff_series.diff().dropna()
            d += 1
        
        self.last_actual_value = series.iloc[-1]
        
        # Phase B: Optimization
        best_cfg = None
        best_score = float('inf')
        
        # Simple grid search for demonstration (in practice pmdarima is used)
        # Using AIC/BIC based on sample size
        criterion = 'bic' if len(series) <= 100 else 'aic'
        
        for p in p_range:
            for q in q_range:
                try:
                    if is_seasonal:
                        model = SARIMAX(series, order=(p, d, q), 
                                        seasonal_order=(1, 0, 1, self.seasonal_period))
                    else:
                        model = ARIMA(series, order=(p, d, q))
                    
                    fit = model.fit(disp=False)
                    score = getattr(fit, criterion)
                    
                    if score < best_score:
                        best_score = score
                        best_cfg = (p, d, q)
                        self.model_fit = fit
                except:
                    continue
        
        if best_cfg is None:
            # Fallback to simple mean forecast if everything else fails
            self.best_model = {'order': (0, d, 0), 'seasonal': False, 'criterion': criterion, 'score': float('inf')}
            model = ARIMA(series, order=(0, d, 0))
            self.model_fit = model.fit()
        else:
            self.best_model = {'order': best_cfg, 'seasonal': is_seasonal, 'criterion': criterion, 'score': best_score}
            
        return self.best_model

    def forecast(self, horizon=24):
        """Recursive Forecasting."""
        if not self.model_fit:
            raise ValueError("Model not selected.")
        
        forecast = self.model_fit.get_forecast(steps=horizon)
        return forecast.summary_frame()

    def diagnostics(self):
        """Model Explainability Dashboard."""
        if not self.model_fit:
            raise ValueError("Model not fitted.")
        
        resid = self.model_fit.resid
        lb_res = acorr_ljungbox(resid, lags=[10], return_df=True)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Residuals", "ACF", "Distribution", "Actual vs Fit"))
        
        # Residuals
        fig.add_trace(go.Scatter(y=resid, name="Residuals"), row=1, col=1)
        
        # ACF
        acf_vals = acf(resid, nlags=20)
        fig.add_trace(go.Bar(y=acf_vals, name="ACF"), row=1, col=2)
        
        # Dist
        fig.add_trace(go.Histogram(x=resid, name="Error Dist"), row=2, col=1)
        
        # Actual vs Fit
        fig.add_trace(go.Scatter(y=self.data[self.target_col], name="Actual"), row=2, col=2)
        fig.add_trace(go.Scatter(y=self.model_fit.fittedvalues, name="Fitted"), row=2, col=2)
        
        fig.update_layout(height=700, title_text=f"Diagnostics: p-value={lb_res['lb_pvalue'].values[0]:.4f}")
        return fig
