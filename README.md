# CX4240_Weather_and_Stocks

Short-term stock return modeling with technical indicators, plus a weather-augmented extension. This repo includes data prep, modeling, evaluation, simple trading rules, portfolio simulation, and backtesting.

Data sources
- Stocks: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data
- NOAA climate: https://www.ncei.noaa.gov/cdo-web/datatools and https://www.ncdc.noaa.gov/cdo-web/results
- US weather (optional): https://www.kaggle.com/datasets/nachiketkamod/weather-dataset-us

Environment
- Python 3.9+
- Install once (top cells in the notebook provide this): backtesting, ta, seaborn, xgboost, shap, scikit-learn, pandas, numpy, matplotlib

Project structure (relevant)
- standard_staock.ipynb — stock-only workflow (feature engineering, models, evaluation, trading, backtesting)
- clean_data/ — per-symbol CSVs with engineered indicators (created by the notebook)
- merged_data.csv — merged dataset (created by the notebook)
- weather_api and weather_stock_analysis notebooks — weather extension (run after stock-only prep)

How to reproduce the stock-only workflow
1) Open standard_staock.ipynb and run cells top-to-bottom (variables are reused).
2) Data + feature engineering
   - Loads S&P 500 OHLCV and computes indicators: EMA(15/50), MACD(+signal), Bollinger Bands, RSI, Stochastic (K/D), ATR(14), OBV, VWAP, ROC(10).
   - Adds engineered features: Target (next-day up/down), Clopen (Close/Open), HighLow (High/Low), log returns (Log5/15/30), daily Pct_Change.
   - Writes per-symbol CSVs to clean_data and creates merged_data.csv.
3) Modeling
   - Decision Tree (classification): Predicts Target; grid-searched hyperparameters. Best found params in notebook: max_depth=20, min_samples_leaf=4, min_samples_split=10, max_features=None. Test accuracy around 0.61 with reduced overfitting vs the no-tuning baseline.
   - Random Forest (classification): Comparable or slightly lower test accuracy depending on depth/features; deep trees can overfit.
   - Linear Regression (regression): Predicts Pct_Change. Evaluated with MSE/MAE/RMSE/R²; scatter and residual plots suggest a weak but usable signal.
   - Coefficient-based feature selection: retrain on top-N absolute coefficients and re-evaluate.
4) Per-stock evaluation
   - Visual arrows over price for recent ~100 days (e.g., GOOG, MRK) indicating predicted next-day move.
   - Cumulative actual vs predicted returns over the last ~1000 days for selected tickers (MRK, PEG, GOOG).
5) XGBoost Regressor
   - Trains XGBRegressor (e.g., n_estimators=100, learning_rate=0.05, max_depth=5).
   - Plots feature importances, computes CV RMSE/MAE/R², and uses SHAP to interpret drivers.
6) Simple trading rules and portfolio simulation
   - Threshold rule: buy a stock on a day if predicted next-day return exceeds a threshold (e.g., 0.0%, 0.1%, 0.3%, 0.5%).
   - Simulates per-symbol equity curves over the last year across ~172 stocks; reports number of winners/losers/no-trades and aggregate P&L. Use the notebook to print exact numbers.
7) Backtesting (GOOG example)
   - Creates a Backtesting.py Strategy that buys when Prediction > 0.05% and exits when Prediction < -0.05%, with $10k cash and 0.2% commission.
   - Runs on the last ~1000 GOOG days and plots the equity curve to compare against buy-and-hold.

Notes and tips
- Run cells in order; paths to clean_data and merged_data are set in the notebook.
- To make results reproducible, set random_state in train_test_split and models.
- Some cells plot many lines (portfolio). Zoom or filter tickers as needed.
- SHAP requires reasonably sized samples; use a subset if performance is slow.

Weather-augmented workflow (optional)
- Run the weather API notebook to fetch/prepare weather features and merge with stock data.
- Then run weather_stock_analysis.ipynb to train and evaluate weather-augmented models. Data access may require an NOAA token and/or Kaggle credentials.

Replicating results
- Exact metrics (accuracy/RMSE/R², trading P&L) are printed in the notebook output and depend on random splits and thresholds; re-run cells to reproduce.

Getting started
- Download the repo ZIP (includes sample data folders and notebooks), open standard_staock.ipynb, install dependencies in the first cell, and execute sequentially.

Questions or access
- This README reflects the methodology and outputs present in standard_staock.ipynb. If you want the README to summarize additional .py/.ipynb files (e.g., the full weather notebooks) or external data not currently in the repo, provide access and I can incorporate them.
