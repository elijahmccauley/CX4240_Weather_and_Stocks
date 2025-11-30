# CX4240 Weather and Stocks

Short-term stock return modeling with technical indicators, plus a weather-augmented extension. This repo includes data prep, modeling, evaluation, simple trading rules, portfolio simulation, and backtesting.

Data sources
- Stocks: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
- NOAA climate: https://www.ncei.noaa.gov/cdo-web/datatools and https://www.ncdc.noaa.gov/cdo-web/results
- US weather (optional): https://www.kaggle.com/datasets/nachiketkamod/weather-dataset-us
- Open-Meteo archive API (used in weather_api_call): https://archive-api.open-meteo.com/v1/archive

Environment
- Python 3.9+
- Install once (top cells in the notebooks provide this): backtesting, ta, seaborn, xgboost, shap, scikit-learn, pandas, numpy, matplotlib, openmeteo-requests, requests-cache, retry-requests

Project structure (relevant)
- standard_staock.ipynb — stock-only workflow (feature engineering, models, evaluation, trading, backtesting)
- weather_api_call.ipynb — fetch and export daily/hourly weather CSVs for New_York, Chicago, Los_Angeles
- weather-stock-analysis.ipynb — merge stock + weather, seasonality features, model comparison, CV, trading sims, SMA benchmark, reporting
- clean_data/ — per-symbol CSVs with engineered indicators (created by the stock notebook)
- merged_data.csv — merged stock-only dataset (created by the stock notebook)
- weather_data/ — outputs and plots from the weather workflows

How to reproduce the stock-only workflow
1) Open standard_staock.ipynb and run cells top-to-bottom (variables are reused).
2) Data + feature engineering
   - Loads S&P 500 OHLCV and computes indicators: EMA(15/50), MACD(+signal), Bollinger Bands, RSI, Stochastic (K/D), ATR(14), OBV, VWAP, ROC(10).
   - Adds engineered features: Target (next-day up/down), Clopen (Close/Open), HighLow (High/Low), log returns (Log5/15/30), daily Pct_Change.
   - Writes per-symbol CSVs to clean_data and creates merged_data.csv.
3) Modeling
   - Decision Tree: Grid-searched hyperparameters; best: max_depth=20, min_samples_leaf=4, min_samples_split=10, max_features=None. Test accuracy ~0.61.
   - Random Forest: Similar or slightly lower test accuracy; watch overfitting.
   - Linear Regression: Predicts Pct_Change; weak but usable signal. Feature selection via coefficients.
4) Per-stock evaluation: arrows over recent days; cumulative actual vs predicted returns over ~1000 days for sample tickers.
5) XGBoost Regressor: feature importances, SHAP interpretation, CV metrics.
6) Threshold trading and portfolio simulation: run over last-year data for ~172 stocks at multiple thresholds (0.0%, 0.1%, 0.3%, 0.5%).
7) Backtesting (GOOG): simple strategy on last ~1000 days with $10k cash, 0.2% commission; plot equity vs buy-and-hold.

Weather workflow (critical)
1) Fetch weather data
   - Open weather_api_call.ipynb. Install openmeteo-requests and its deps. Run to export hourly and daily CSVs per city and combined files into weather_data/.
   - Cities: New_York, Chicago, Los_Angeles; daily features include temperature, wind, gusts, precipitation, snowfall, daylight, sunshine, hours with precip.
2) Load and prepare stock + weather
   - Open weather-stock-analysis.ipynb and run sequentially.
   - Load clean_data/*.csv; ensure Ticker column exists; parse Date.
   - Load weather_data/combined_daily_weather_data.csv; select New_York as proxy weather; rename date->Date.
3) Seasonality features
   - Weather: month, day_of_year, season (1-4), season_name; seasonal averages and deviations; seasonal z-scores; extreme anomalies flags; engineered features (temp range, extreme heat/cold, heavy_rain, snow_day, strong_wind, sunshine_ratio); lags (1/2/3) and rolling means/std; fill NA via bfill/ffill.
   - Stocks: month, day_of_year, quarter, year, weekday flags; seasonality markers (January/December/October effects, summer, quarter_end, Monday/Friday).
4) Merge by date-only
   - Create Date_Only for both; filter to overlapping dates; aggregate weather to daily; merge into combined_df; ensure season and season_name preserved.
5) Feature sets
   - Define base_features (stock + stock seasonality) and weather_only_features (weather + weather seasonality). Exclude Date/Ticker/Symbol/strings. X_base and X_all are numeric-only.
6) Model comparison (classification of Target)
   - Train RandomForest (n_estimators=100, max_depth=20, min_leaf=4, min_split=10, random_state=42) on base and combined features. Scale inputs with StandardScaler.
   - Report metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, log loss, brier; confusion matrices saved to weather_data/.
   - Plot top feature importances; list weather features appearing in top 20.
   - ROC comparison plot saved.
7) Season-specific models
   - Evaluate models per season (Winter/Spring/Summer/Fall) when sufficient samples; save seasonal feature importance plots; summarize accuracy/F1/ROC-AUC per season.
8) Time series cross-validation
   - TimeSeriesSplit (n_splits=3) with lighter RF params; optional stratified sampling for large datasets; report avg accuracy for base vs combined and overall improvement; save CV plots and metrics comparisons; analyze seasonal improvement per fold.
9) Stock-specific impact
   - For top ~20 tickers with enough data, compare base vs combined model per season; derive per-ticker seasonal weather impact; plot top stock-season combinations.
10) Trading simulations
   - Enhanced simulation: signal from model probability (>0.7 buy, <0.3 sell), slippage and transaction costs, track position, value, returns, drawdowns, Sharpe, win-rate, and trades; save top-ticker performance plots and portfolio comparisons.
   - Fixed weather strategy: combined signals (base + weather) with weather extremity features; compare to base and buy-and-hold; save per-ticker charts and summary.
   - SMA benchmark: 20/50-day SMA crossover with costs; compare returns and Sharpe to weather strategy, base, and buy‑and‑hold; append summary to weather_impact_report.md.
11) Reporting
   - weather_data/weather_impact_report.md is generated with overall improvement, top weather features, strongest seasonal effects, top stock-season pairs, and strategy comparison.

Key findings
- Adding weather features often improves classification metrics vs stock-only baselines, with variability by season and ticker.
- Certain weather features (e.g., precipitation_sum, temperature_2m_mean deviations) appear among top importances in combined models.
- Seasonal effects matter: improvement tends to be non-uniform across Winter/Summer.
- Simple trading signals from model confidences with costs can show differences between stock-only and stock+weather; the SMA benchmark provides a baseline.

Notes and tips
- Run cells in order in both notebooks; many variables are reused.
- Ensure weather_data CSVs exist before running merge steps.
- Set random_state in train/test splits and models to stabilize metrics.
- Some plots and simulations are heavy; filter to fewer tickers or sample data for speed.

Getting started
- Download the repo ZIP, open standard_staock.ipynb, install dependencies, and run sequentially to create clean_data/ and merged_data.csv.
- Then open weather_api_call.ipynb to create weather_data/ CSVs.
- Finally run weather-stock-analysis.ipynb for merges, modeling, CV, trading sims, SMA benchmark, and reporting.
