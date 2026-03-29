import torch
from extensions import TimeSeriesTransformerLSTM, ProbabilisticForecaster, prepare_tensors
import numpy as np

def test_extensions():
    print("Testing Hybrid Transformer/LSTM...")
    model = TimeSeriesTransformerLSTM(input_dim=1, model_dim=64, num_heads=4, num_layers=2, output_dim=1)
    x = torch.randn(32, 12, 1) # batch, seq, dim
    out = model(x)
    print(f"Hybrid model output shape: {out.shape}")
    assert out.shape == (32, 1)

    print("Testing Probabilistic Forecaster...")
    pf = ProbabilisticForecaster()
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    pf.fit(X, y)
    y_pred, sigma = pf.predict(X[:5])
    print(f"Probabilistic predictions shape: {y_pred.shape}, sigma shape: {sigma.shape}")
    assert len(y_pred) == 5

if __name__ == "__main__":
    test_extensions()
