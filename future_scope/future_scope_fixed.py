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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
import torch
import torch.nn as nn
import warnings
import time

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
        self.ensemble_models = []
        self.is_seasonal = False
        
        # Extensions state
        self.gp_regressor = None
        self.transformer_model = None

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

    def preprocess(self, freq=None, light_mode=False, skip_outliers=False, skip_transform=False):
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
        
        # 2. Duplicate Resolution
        df = df[~df.index.duplicated(keep='first')]
        
        if not light_mode:
            # 3. Scalable Outlier Management
            if not skip_outliers:
                n_samples = len(df)
                series = df[self.target_col]
                
                if n_samples < 10000:
                    # Small Dataset Logic
                    _, p_val = stats.shapiro(series.dropna()) if n_samples <= 5000 else (0, 0)
                    skewness = stats.skew(series.dropna())
                    
                    if p_val > 0.05 and abs(skewness) < 0.5:
                        z_scores = np.abs(stats.zscore(series))
                        outlier_mask = z_scores > 3
                    else:
                        q1, q3 = series.quantile([0.25, 0.75])
                        iqr = q3 - q1
                        outlier_mask = (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
                    
                    if n_samples < 5000:
                        df.loc[outlier_mask, self.target_col] = series.median()
                    else:
                        lower = series.quantile(0.05)
                        upper = series.quantile(0.95)
                        df[self.target_col] = np.clip(series, lower, upper)
                else:
                    iso = IsolationForest(contamination=0.05, random_state=42)
                    preds = iso.fit_predict(series.values.reshape(-1, 1))
                    outlier_mask = preds == -1
                    df.loc[outlier_mask, self.target_col] = series.median()

            # 4. Non-Linear Corrections
            if not skip_transform:
                skewness = stats.skew(df[self.target_col])
                if abs(skewness) > 1.0:
                    pt = PowerTransformer(method='yeo-johnson')
                    df[self.target_col] = pt.fit_transform(df[[self.target_col]]).flatten()
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
        return res[1] < 0.05

    def select_model(self, mode='ensemble', max_order=5):
        """Enhanced Model Selection with Ensemble and stepwise search.

        Args:
            mode: 'ensemble' for ensemble, 'simple' for single best model
            max_order: Maximum ARIMA order for search (default 5, use 3 for faster computation)
        """
        series = self.data[self.target_col]

        # Phase A: Seasonal Strength
        stl_res = self.decompose(method='stl')
        self.stl_res = stl_res # Save for hybrid extension
        seasonal_strength = max(0, 1 - np.var(stl_res.resid) / np.var(stl_res.seasonal + stl_res.resid))

        self.is_seasonal = seasonal_strength >= 0.1
        self.history['seasonal_strength'] = seasonal_strength

        print(f"Selecting model (is_seasonal={self.is_seasonal}, strength={seasonal_strength:.4f})...")

        # Phase B: Optimization using pmdarima for robust search
        # Use reduced max_order for faster computation in benchmarks
        model = pm.auto_arima(series,
                              seasonal=self.is_seasonal,
                              m=self.seasonal_period if self.is_seasonal else 1,
                              start_p=0, max_p=max_order,
                              start_q=0, max_q=max_order,
                              d=None, max_d=2,
                              max_order=max_order * 2,  # Combined p+q+P+Q
                              stepwise=True,
                              trace=False,  # Disable verbose output for cleaner logs
                              suppress_warnings=True,
                              error_action='ignore',
                              n_jobs=1)  # Single thread to avoid overhead

        self.best_model = {'order': model.order, 'seasonal_order': model.seasonal_order}

        if mode == 'ensemble':
            # Simplified Ensemble: Top-3 models by BIC
            # In a real scenario we'd run multiple trials, here we use pmdarima's best as base
            # and fit 2 alternative variations (p-1, q+1) etc.
            p, d, q = model.order
            configs = [(p, d, q), (max(0, p-1), d, q), (p, d, max(0, q-1))]
            self.ensemble_models = []
            for cfg in set(configs):
                try:
                    if self.is_seasonal:
                        m = SARIMAX(series, order=cfg, seasonal_order=model.seasonal_order).fit(disp=False)
                    else:
                        m = ARIMA(series, order=cfg).fit(disp=False)
                    self.ensemble_models.append(m)
                except:
                    continue
            self.model_fit = self.ensemble_models[0] # primary for diagnostics
        else:
            self.model_fit = model.arima_res_

        return self.best_model

    def forecast(self, horizon=24, use_ensemble=True):
        """Recursive Forecasting with Ensemble Averaging."""
        if not self.model_fit and not self.ensemble_models:
            raise ValueError("Model not selected.")
        
        if use_ensemble and self.ensemble_models:
            forecasts = []
            for m in self.ensemble_models:
                forecasts.append(m.get_forecast(steps=horizon).summary_frame()['mean'])
            ensemble_mean = pd.concat(forecasts, axis=1).mean(axis=1)
            # Standard CI from the best model
            summary = self.model_fit.get_forecast(steps=horizon).summary_frame()
            summary['mean'] = ensemble_mean
            return summary
        else:
            forecast = self.model_fit.get_forecast(steps=horizon)
            return forecast.summary_frame()

    # --- Extensions ---

    def fit_gp_uncertainty(self):
        """Extension: Bayesian GP Uncertainty on residuals."""
        resid = self.model_fit.resid
        X = np.arange(len(resid)).reshape(-1, 1)
        y = resid.values
        
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gp_regressor = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5)
        self.gp_regressor.fit(X, y)
        print("Gaussian Process fitted on residuals.")

    def forecast_with_gp(self, horizon=24):
        """Forecast combining ARIMA mean + GP uncertainty."""
        base_fc = self.forecast(horizon)
        X_future = np.arange(len(self.data), len(self.data) + horizon).reshape(-1, 1)
        gp_mean, gp_std = self.gp_regressor.predict(X_future, return_std=True)
        
        base_fc['mean'] += gp_mean
        base_fc['mean_ci_lower'] = base_fc['mean'] - 1.96 * gp_std
        base_fc['mean_ci_upper'] = base_fc['mean'] + 1.96 * gp_std
        return base_fc

    def fit_hybrid_transformer(self, epochs=10):
        """Extension: Hybrid STL-Transformer."""
        resid = self.stl_res.resid.dropna().values
        input_dim = 1
        model_dim = 32
        nhead = 2
        num_layers = 2
        
        class TinyTransformer(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead, batch_first=True)
                self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
                self.fc = nn.Linear(model_dim, 1)
                self.embedding = nn.Linear(input_dim, model_dim)

            def forward(self, x):
                x = self.embedding(x)
                x = self.transformer(x)
                return self.fc(x[:, -1, :])

        self.transformer_model = TinyTransformer()
        optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Prepare data (lag 12)
        X, y = [], []
        for i in range(len(resid) - 12):
            X.append(resid[i:i+12])
            y.append(resid[i+12])
        X, y = torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
        
        self.transformer_model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            out = self.transformer_model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        print("Hybrid Transformer fitted.")

    def forecast_hybrid(self, horizon=24):
        """Forecast combining STL Trend/Seasonal + Transformer residuals."""
        # Future Trend/Seasonal (simplified as persistence/extension)
        res = self.stl_res
        last_trend = res.trend.iloc[-1]
        last_seasonal = res.seasonal.iloc[-self.seasonal_period:].values
        
        # Transformer residual forecast (recursive)
        self.transformer_model.eval()
        curr_seq = torch.FloatTensor(self.stl_res.resid.tail(12).values).reshape(1, 12, 1)
        resid_fcs = []
        for _ in range(horizon):
            with torch.no_grad():
                pred = self.transformer_model(curr_seq).item()
                resid_fcs.append(pred)
                # update sequence
                new_val = torch.FloatTensor([[[pred]]])
                curr_seq = torch.cat([curr_seq[:, 1:, :], new_val], dim=1)
        
        # Combine
        combined = []
        for i in range(horizon):
            s = last_seasonal[i % self.seasonal_period]
            combined.append(last_trend + s + resid_fcs[i])
        
        fc_df = pd.DataFrame({'mean': combined}, index=pd.date_range(start=self.data.index[-1] + self.data.index.freq, periods=horizon, freq=self.freq))
        return fc_df

    def diagnostics(self):
        """Model Explainability Dashboard."""
        if not self.model_fit:
            raise ValueError("Model not fitted.")
        
        resid = self.model_fit.resid
        lb_res = acorr_ljungbox(resid, lags=[10], return_df=True)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Residuals", "ACF", "Distribution", "Actual vs Fit"))
        fig.add_trace(go.Scatter(y=resid, name="Residuals"), row=1, col=1)
        acf_vals = acf(resid, nlags=20)
        fig.add_trace(go.Bar(y=acf_vals, name="ACF"), row=1, col=2)
        fig.add_trace(go.Histogram(x=resid, name="Error Dist"), row=2, col=1)
        fig.add_trace(go.Scatter(y=self.data[self.target_col], name="Actual"), row=2, col=2)
        fig.add_trace(go.Scatter(y=self.model_fit.fittedvalues, name="Fitted"), row=2, col=2)
        fig.update_layout(height=700, title_text=f"Diagnostics: p-value={lb_res['lb_pvalue'].values[0]:.4f}")
        return fig
