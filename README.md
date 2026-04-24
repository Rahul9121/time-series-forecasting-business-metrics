# Time Series Forecasting for Business Metrics
Interactive forecasting dashboard for business sales/demand trends with **ARIMA vs ML comparison**, seasonality diagnostics, and exportable forecasts.

## Live App
https://time-series-forecasting-business-metrics-by-rahulega.streamlit.app/

## About
This project was built to simulate a real business forecasting workflow where you need to:
- understand historical sales behavior,
- detect trend/seasonality patterns,
- compare multiple forecasting approaches,
- and generate practical forward forecasts for planning decisions.

The dashboard is branded with: **rahul ega**.

## Core Features
- Weekly time-series forecasting from Walmart sales data
- Seasonality detection (autocorrelation + decomposition)
- Model benchmarking:
  - **ARIMA**
  - **ML (RandomForest with lag features)**
  - **Ensemble forecast (ARIMA + ML average)**
- Interactive controls for:
  - aggregation level (global/store/department/store+department)
  - holdout window
  - forecast horizon
  - ARIMA parameters
  - ML tree count
- Forecast CSV export for downstream analysis

## Dataset
**Walmart Store Sales Forecasting**
- Competition source: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
- Public mirror used by app runtime:
  - `train.csv`
  - `features.csv`
  - `stores.csv`
  from: `https://raw.githubusercontent.com/gagandeepsinghkhanuja/Walmart-Sales-Forecasting/master`

## Tech Stack
- Python
- Pandas / NumPy
- statsmodels (ARIMA/SARIMA)
- scikit-learn (RandomForest)
- Streamlit
- Docker

## Project Structure
- `app.py` - Streamlit UI and orchestration
- `src/forecasting/data.py` - data ingestion/prep
- `src/forecasting/seasonality.py` - seasonality detection
- `src/forecasting/models.py` - ARIMA/ML/ensemble logic
- `requirements.txt` - dependencies
- `Dockerfile` - containerized run setup

## Run Locally
1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```
4. Open:
   `http://localhost:8501`

## Run with Docker
Build image:
```bash
docker build -t business-forecasting-dashboard .
```
Run container:
```bash
docker run -p 8501:8501 business-forecasting-dashboard
```

## Deploy to Streamlit Community Cloud
1. Push this repository to GitHub.
2. Go to https://share.streamlit.io
3. Sign in with GitHub.
4. Click **New app**.
5. Select:
   - Repository: this repo
   - Branch: `main`
   - Main file path: `app.py`
6. Click **Deploy**.

## Use Cases
- Demand planning
- Revenue trend monitoring
- Inventory readiness planning
- Promotion impact review (with seasonal context)
