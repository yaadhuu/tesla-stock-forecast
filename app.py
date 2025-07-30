from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
from arch import arch_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Tesla Stock Forecast API is Running"}

@app.get("/forecast")
def forecast():
    ticker = 'TSLA'
    start_date = '2018-01-01'
    end_date = '2025-01-01'
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    data.reset_index(inplace=True)
    data.columns = [col[1] if isinstance(col, tuple) else col for col in data.columns.values]

    df = data[['Date', 'Close']].rename(columns={'Close': 'Close'})
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)

    df_prophet = df[['Close']].reset_index().rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet_model = Prophet(daily_seasonality=True)
    prophet_model.fit(df_prophet)
    future = prophet_model.make_future_dataframe(periods=60)
    forecast_prophet = prophet_model.predict(future)

    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df[['Close']])
    train_size = int(len(scaled_close) * 0.8)
    train_data = scaled_close[:train_size]
    test_data = scaled_close[train_size - 60:]

    def create_sequences(data, time_steps=60):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i - time_steps:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data)
    X_test, y_test = create_sequences(test_data)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)
    lstm_pred_scaled = lstm_model.predict(X_test)
    lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()

    series = df['Close']
    arima_model = ARIMA(series, order=(5,1,0)).fit()
    arima_forecast = arima_model.forecast(steps=60)

    sarima_model = SARIMAX(series, order=(2,1,1), seasonal_order=(1,1,1,7)).fit(disp=False)
    sarima_forecast = sarima_model.forecast(steps=60)

    returns = 100 * df['Close'].pct_change().dropna()
    garch_model = arch_model(returns, vol='Garch', p=1, q=1)
    garch_fit = garch_model.fit(disp='off')
    garch_forecast = garch_fit.forecast(horizon=60)
    volatility_forecast = np.sqrt(garch_forecast.variance.values[-1, :]).tolist()

    rolling_mean = df['Close'].rolling(window=30).mean()
    rolling_std = df['Close'].rolling(window=30).std()
    df['Z-Score'] = (df['Close'] - rolling_mean) / rolling_std
    df['Anomaly'] = df['Z-Score'].abs() > 2.5
    anomalies = df[df['Anomaly']].tail(10)[['Close']].reset_index().to_dict(orient='records')

    result = {
        "LSTM_Forecast": lstm_pred[-10:].tolist(),
        "ARIMA_Forecast": arima_forecast.values[-10:].tolist(),
        "SARIMA_Forecast": sarima_forecast.values[-10:].tolist(),
        "Prophet_Forecast": forecast_prophet['yhat'].iloc[-10:].tolist(),
        "GARCH_Volatility": volatility_forecast[-10:],
        "Anomalies": anomalies
    }

    return result
