# 🚀 Tesla Stock Forecasting API 📊🔮

![Tesla Forecast Banner](https://img.shields.io/badge/Tesla-Stock%20Prediction-red?style=for-the-badge&logo=tesla&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-⚡-green?style=for-the-badge&logo=fastapi)
![Render Deployment](https://img.shields.io/badge/Deployed%20on-Render-blue?style=for-the-badge&logo=render)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python)

---

## 📈 About the Project
This project provides a **REST API** that forecasts **Tesla (TSLA) stock prices** using advanced **Time Series Forecasting Models**:
- 🧠 **LSTM (Long Short-Term Memory) Neural Network**
- 📈 **ARIMA & SARIMA Statistical Models**
- 🔮 **Facebook Prophet Forecasting**
- 📊 **GARCH for Volatility Prediction**
- 🚨 **Anomaly Detection (Z-score Method)**

Deployed seamlessly on **Render Cloud Platform** using **FastAPI** as the backend framework.

---

## 🌐 Live Demo
| Endpoint | URL |
|----------|-----|
| **Base URL** | `https://<your-render-url>.onrender.com/` |
| **Forecast API** | `https://<your-render-url>.onrender.com/forecast` |

---

## 🏗️ Tech Stack
| Backend | Forecasting Models | Deployment |
|---------|--------------------|------------|
| ![FastAPI](https://img.shields.io/badge/FastAPI-⚡-green?style=flat-square&logo=fastapi) | LSTM, ARIMA, SARIMA, Prophet, GARCH | ![Render](https://img.shields.io/badge/Render-Cloud-blue?style=flat-square&logo=render) |
| Python 3.10+ | Z-score Anomaly Detection | GitHub Integration |

---

## 📊 API Response Example
```json
{
  "LSTM_Forecast": [280.45, 282.32, 283.11, ...],
  "ARIMA_Forecast": [279.11, 280.54, 282.76, ...],
  "SARIMA_Forecast": [278.90, 280.40, 281.70, ...],
  "Prophet_Forecast": [280.12, 281.89, 283.01, ...],
  "GARCH_Volatility": [2.12, 2.15, 2.18, ...],
  "Anomalies": [
    {"Date": "2024-09-15", "Close": 350.75},
    {"Date": "2024-10-03", "Close": 360.22}
  ]
}
