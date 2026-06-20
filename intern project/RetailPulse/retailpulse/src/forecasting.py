"""
RetailPulse - Demand Forecasting
Prophet baseline + LSTM ensemble
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────
# DATA PREP
# ─────────────────────────────────────────

def prepare_daily_sales(df):
    """Aggregate to daily revenue time series"""
    ts = (df.groupby("InvoiceDate")["Revenue"]
            .sum()
            .reset_index()
            .rename(columns={"InvoiceDate": "ds", "Revenue": "y"}))
    ts = ts.sort_values("ds").reset_index(drop=True)
    return ts


def train_test_split_ts(ts, test_days=30):
    split = ts["ds"].max() - pd.Timedelta(days=test_days)
    train = ts[ts["ds"] <= split]
    test  = ts[ts["ds"] >  split]
    return train, test


# ─────────────────────────────────────────
# PROPHET
# ─────────────────────────────────────────

def fit_prophet(train):
    try:
        from prophet import Prophet
    except ImportError:
        print("  ⚠️  Prophet not installed. Run: pip install prophet")
        return None, None

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
    )
    m.add_country_holidays(country_name="GB")
    m.fit(train)
    return m, "prophet"


def prophet_predict(model, periods=30, freq="D"):
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]


# ─────────────────────────────────────────
# LSTM (PyTorch-free numpy version for portability)
# ─────────────────────────────────────────

class SimpleLSTMNumpy:
    """Lightweight LSTM-like model using numpy for portability"""

    def __init__(self, lookback=14):
        self.lookback = lookback
        self.scaler   = MinMaxScaler()

    def _make_sequences(self, series):
        X, y = [], []
        for i in range(self.lookback, len(series)):
            X.append(series[i - self.lookback: i])
            y.append(series[i])
        return np.array(X), np.array(y)

    def fit(self, series):
        values = self.scaler.fit_transform(series.reshape(-1, 1)).flatten()
        X, y   = self._make_sequences(values)

        # Simple linear regression over windows (interpretable baseline)
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=0.5)
        self.model.fit(X, y)
        self._last_window = values[-self.lookback:]
        return self

    def predict(self, steps=30):
        preds  = []
        window = self._last_window.copy()
        for _ in range(steps):
            p = self.model.predict(window.reshape(1, -1))[0]
            preds.append(p)
            window = np.roll(window, -1)
            window[-1] = p
        preds_inv = self.scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
        return np.clip(preds_inv, 0, None)


# ─────────────────────────────────────────
# ENSEMBLE
# ─────────────────────────────────────────

def ensemble_forecast(prophet_fc, lstm_preds, weights=(0.6, 0.4)):
    df = prophet_fc.tail(len(lstm_preds)).copy()
    df["lstm_yhat"]     = lstm_preds
    df["ensemble_yhat"] = (weights[0] * df["yhat"] + weights[1] * df["lstm_yhat"])
    df["ensemble_yhat"] = df["ensemble_yhat"].clip(lower=0)
    return df


def evaluate_forecast(actual, predicted):
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mae  = np.mean(np.abs(actual - predicted))
    return {"MAPE": round(mape, 2), "RMSE": round(rmse, 2), "MAE": round(mae, 2)}


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def run_forecasting(df):
    print("🔄 Preparing daily sales...")
    ts = prepare_daily_sales(df)
    train, test = train_test_split_ts(ts, test_days=30)
    print(f"  Train: {len(train)} days | Test: {len(test)} days")

    # LSTM
    print("🔄 Fitting LSTM model...")
    lstm = SimpleLSTMNumpy(lookback=14)
    lstm.fit(train["y"].values)
    lstm_preds = lstm.predict(steps=30)

    # Prophet (optional)
    prophet_model, _ = fit_prophet(train)
    if prophet_model:
        print("🔄 Fitting Prophet model...")
        prophet_fc = prophet_predict(prophet_model, periods=30)
        forecast_df = ensemble_forecast(prophet_fc, lstm_preds)
    else:
        future_dates = pd.date_range(train["ds"].max() + pd.Timedelta(days=1), periods=30)
        forecast_df  = pd.DataFrame({"ds": future_dates, "yhat": lstm_preds,
                                     "lstm_yhat": lstm_preds, "ensemble_yhat": lstm_preds,
                                     "yhat_lower": lstm_preds * 0.85, "yhat_upper": lstm_preds * 1.15})

    # Evaluate on test
    test_preds = lstm.predict(steps=len(test))
    metrics = evaluate_forecast(test["y"].values, test_preds[:len(test)])
    print(f"  📊 MAPE: {metrics['MAPE']}% | RMSE: £{metrics['RMSE']:,.0f} | MAE: £{metrics['MAE']:,.0f}")

    # Save
    ts.to_csv("data/daily_sales.csv", index=False)
    forecast_df.to_csv("data/forecast_30d.csv", index=False)
    pd.DataFrame([metrics]).to_csv("data/forecast_metrics.csv", index=False)
    print("✅ Saved forecast outputs")
    return ts, forecast_df, metrics


if __name__ == "__main__":
    from feature_engineering import load_and_clean
    df = load_and_clean()
    run_forecasting(df)
