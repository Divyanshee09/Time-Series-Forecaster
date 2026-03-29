import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

class TimeSeriesTransformerLSTM(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim):
        super(TimeSeriesTransformerLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, model_dim, batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(model_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, model_dim)
        trans_out = self.transformer(lstm_out.transpose(0, 1))
        # trans_out: (seq_len, batch, model_dim)
        out = self.fc(trans_out[-1, :, :])
        return out

class ProbabilisticForecaster:
    def __init__(self):
        self.gpr = None

    def fit(self, X, y):
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
        self.gpr.fit(X, y)

    def predict(self, X):
        y_pred, sigma = self.gpr.predict(X, return_std=True)
        return y_pred, sigma

def prepare_tensors(series, n_lags=12):
    X, y = [], []
    for i in range(len(series) - n_lags):
        X.append(series[i:i+n_lags])
        y.append(series[i+n_lags])
    return torch.FloatTensor(np.array(X)).unsqueeze(-1), torch.FloatTensor(np.array(y)).unsqueeze(-1)
