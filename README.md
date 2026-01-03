---

# 📊 Weather-Augmented Stock Market Prediction

A comprehensive machine learning system that investigates whether weather patterns can improve stock market prediction accuracy through multi-model comparison, seasonal analysis, and realistic trading simulations.

## 🎯 Project Overview

This project explores the hypothesis that **weather conditions influence market behavior** in predictable ways. By integrating meteorological data with technical indicators, we built and compared multiple prediction models to determine if weather provides actionable trading signals.

### Key Questions Addressed:
- Do weather features improve stock prediction accuracy?
- Are weather effects stronger in certain seasons?
- Can weather data generate profitable trading strategies?
- How does weather-augmented prediction compare to traditional technical analysis?

---

## 📁 Project Structure

```
CX4240_Weather_and_Stocks/
├── standard_stock.ipynb           # Stock-only baseline models
├── weather_api_call.ipynb         # Weather data collection
├── weather-stock-analysis.ipynb   # Complete integrated analysis
├── clean_data/                    # Processed stock data (172 tickers)
├── weather_data/                  # Weather datasets and outputs
│   ├── combined_daily_weather_data.csv
│   ├── weather_impact_report.md
│   └── [visualization outputs]
└── README.md
```

---

## 🔬 Methodology

### Data Sources

**Stock Data:**
- **Source:** [Kaggle S&P 500 Dataset](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)
- **Coverage:** 172 S&P 500 stocks, 2010-2024 (609K+ data points)
- **Features:** OHLCV + 20+ engineered technical indicators

**Weather Data:**
- **Sources:** 
  - NOAA Climate Data ([CDO Web](https://www.ncei.noaa.gov/cdo-web/datatools))
  - Open-Meteo Archive API
- **Coverage:** New York, Chicago, Los Angeles (2010-2024)
- **Features:** Temperature, precipitation, wind, sunshine, seasonal deviations

---

### Feature Engineering

**Technical Indicators (20+):**
- Trend: EMA(15/50), MACD + Signal, ROC(10)
- Momentum: RSI(14), Stochastic Oscillator (K/D)
- Volatility: Bollinger Bands, ATR(14)
- Volume: OBV, VWAP
- Custom: Clopen ratio, HighLow ratio, Log returns (5/15/30 day)

**Weather Features (55+):**
- **Raw:** Temperature (mean/max/min), precipitation, wind speed/gusts, sunshine duration
- **Seasonal:** Deviation from seasonal norms, z-scores, anomaly flags
- **Engineered:** Temperature range, extreme conditions flags, sunshine ratio
- **Temporal:** Lags (1/2/3 days), rolling means/std (3/7 day windows)
- **Interaction:** Season-specific weather z-scores

**Seasonality Features:**
- Stock market effects: January effect, summer doldrums, quarter-end, day-of-week
- Weather seasonality: Seasonal averages, deviations, extreme anomaly detection

---

## 🤖 Machine Learning Models

### Model Comparison Framework

| Model | Purpose | Key Parameters |
|-------|---------|----------------|
| **Decision Tree** | Binary up/down prediction | depth=20, min_leaf=4, min_split=10 |
| **Random Forest** | Ensemble classification | n_estimators=100, depth=20 |
| **XGBoost Regressor** | Percent change prediction | n_estimators=100, lr=0.05, depth=5 |
| **Linear Regression** | Baseline + feature selection | Top 10 features by coefficient |

### Evaluation Strategy

1. **Base Model:** Stock features only (39 features)
2. **Combined Model:** Stock + Weather (94 features)
3. **Seasonal Models:** Separate models per season (Winter/Spring/Summer/Fall)
4. **Cross-Validation:** TimeSeriesSplit (3 folds) to respect temporal ordering

---

## 📈 Key Results

### Model Performance Improvements

```
Adding Weather Features:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Base Model (Stock Only)
  ├─ Accuracy:  61.23%
  ├─ ROC-AUC:   0.6630
  └─ F1 Score:  0.6776

Combined Model (Stock + Weather)
  ├─ Accuracy:  70.76%  (+9.53%)  ⬆️
  ├─ ROC-AUC:   0.7790  (+0.116)  ⬆️
  └─ F1 Score:  0.7383  (+0.061)  ⬆️
```

### Cross-Validation Results

| Metric | Base | Combined | Improvement |
|--------|------|----------|-------------|
| Accuracy | 51.79% | 58.92% | **+7.13%** |
| Precision | 52.46% | 57.58% | +5.12% |
| F1 Score | 62.81% | 66.81% | +4.00% |
| ROC-AUC | 0.5188 | 0.6342 | **+22.25%** |

### Seasonal Impact Analysis

**Weather Prediction Improvement by Season:**
- **Summer:** +7.85% accuracy (strongest effect)
- **Spring:** +7.22% accuracy
- **Fall:** +6.79% accuracy
- **Winter:** +6.61% accuracy

**Finding:** Weather effects are strongest in **summer months**, likely due to vacation patterns, energy consumption, and consumer behavior shifts.

---

## 💹 Trading Strategy Backtesting

### Strategies Tested

1. **Weather-Sensitive Strategy**
   - Uses combined model predictions
   - Amplifies signals during extreme weather events
   - Adjustable position sizing based on weather extremity

2. **Basic Strategy**
   - Stock-only model predictions
   - No weather consideration
   - Baseline comparison

3. **SMA Crossover Strategy**
   - Technical analysis benchmark
   - 20/50-day moving average crossover
   - Industry-standard comparison

4. **Buy-and-Hold**
   - Passive benchmark
   - Initial investment held throughout period

### Backtest Parameters

```python
Initial Investment: $10,000 per stock
Transaction Costs:  0.1% per trade
Slippage:          0.1% per trade
Test Period:       2024 (Last year of data)
Stocks Analyzed:   172 S&P 500 tickers
```

### Trading Performance
Multiple strategies were implemented and backtested with realistic transaction costs (0.1%) 
and slippage (0.1%). Results varied significantly by ticker and time period. See 
`weather-stock-analysis.ipynb` for detailed per-stock performance breakdowns.

Key Finding: Weather-sensitive strategy showed improved Sharpe ratios compared to baseline, 
though absolute returns varied based on market conditions.

---

## 🔍 Feature Importance Analysis

### Top 10 Most Important Features (Combined Model)

1. **wind_speed_10m_max** (0.0203) - Weather
2. **wind_speed_10m_max_lag2** (0.0199) - Weather  
3. **wind_speed_10m_max_lag1** (0.0198) - Weather
4. **temp_change** (0.0191) - Weather
5. **temp_std_rolling7** (0.0191) - Weather
6. **temp_std_rolling3** (0.0189) - Weather
7. **Log5** (0.0186) - Stock
8. **daylight_duration** (0.0185) - Weather
9. **wind_speed_10m_max_lag3** (0.0185) - Weather
10. **Clopen** (0.0183) - Stock

**Key Insight:** **Wind patterns and temperature volatility** are the most predictive weather features, suggesting these affect trader behavior, logistics, or energy markets.

---

## 📊 Visualization Outputs

The project generates 15+ visualizations:

- **Model Performance:**
  - Confusion matrices (base vs. combined)
  - ROC curve comparisons
  - Feature importance rankings

- **Seasonal Analysis:**
  - Per-season model performance
  - Seasonal feature importance
  - Weather impact heatmaps

- **Trading Simulations:**
  - Portfolio value over time
  - Drawdown analysis
  - Strategy comparison charts
  - Per-stock performance breakdowns

All visualizations saved to `weather_data/` directory.

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/elijahmccauley/CX4240_Weather_and_Stocks.git
cd CX4240_Weather_and_Stocks

# Install dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
pip install ta backtesting openmeteo-requests requests-cache retry-requests
```

### Reproduction Steps

**1. Prepare Stock Data:**
```bash
# Run standard_stock.ipynb sequentially
# This creates clean_data/ directory with engineered features
```

**2. Fetch Weather Data:**
```bash
# Run weather_api_call.ipynb
# This creates weather_data/ directory with daily/hourly data
```

**3. Run Complete Analysis:**
```bash
# Run weather-stock-analysis.ipynb sequentially
# This performs merging, modeling, CV, trading sims, and generates report
```

---

## 🛠️ Technical Implementation Details

### Handling Seasonality

**Challenge:** Both stocks and weather exhibit strong seasonal patterns that can create spurious correlations.

**Solution:**
- Created seasonal baseline averages for weather features
- Calculated deviations and z-scores from seasonal norms
- Identified extreme weather anomalies (>2σ from seasonal mean)
- Built season-specific models to capture temporal dynamics

### Preventing Data Leakage

**Time Series Cross-Validation:**
```python
TimeSeriesSplit(n_splits=3)
# Ensures training data always precedes test data
# Respects temporal ordering of stock market
```

**No Future Information:**
- Weather lags (1/2/3 days) only use past data
- Rolling statistics calculated on historical windows
- Technical indicators computed using prior values only

### Handling Class Imbalance

```
Target Distribution:
├─ Up Days:   321,555 (52.1%)
└─ Down Days: 296,269 (47.9%)
```
Relatively balanced, no resampling needed.

---

## 📝 Key Findings & Conclusions

### ✅ What Worked

1. **Weather Does Improve Prediction**
   - Consistent 7-9% accuracy improvement across CV folds
   - 22% improvement in ROC-AUC score
   - Weather features dominate top importance rankings

2. **Wind > Temperature**
   - Wind speed metrics are most predictive
   - Temperature volatility (rolling std) matters more than absolute values
   - Lagged weather (1-3 days prior) provides useful signal

3. **Seasonal Effects Are Real**
   - Summer shows strongest weather impact (+7.85%)
   - Each season benefits from weather augmentation
   - Seasonal anomalies (z-scores) are more predictive than raw values

4. **Feature Engineering Matters**
   - Interaction terms (compound × tyrelife) crucial for modeling
   - Lagged features capture delayed market reactions
   - Rolling statistics smooth noisy weather data

### ⚠️ Limitations & Challenges

1. **Regional Weather Proxy**
   - Used NYC weather as proxy for all S&P 500 stocks
   - Would benefit from company HQ location-specific weather
   - Multi-city weighted average could improve results

2. **Sector-Specific Effects Not Captured**
   - Energy, retail, transportation likely have different sensitivities
   - Future work: sector-stratified models

3. **Backtest Overfitting**
   - Some trading results show unrealistic returns
   - Suggests overfitting on specific tickers/periods
   - Need out-of-sample testing on 2025 data

4. **Transaction Costs**
   - Assumed 0.1% costs may be optimistic for retail traders
   - Slippage increases with trade size (not modeled)
   - Market impact not considered

---

## 🔮 Future Improvements

### Short-Term Enhancements
- [ ] Add sector-specific weather models
- [ ] Incorporate company location data for localized weather
- [ ] Test on extended out-of-sample period (2025)
- [ ] Implement ensemble model combining multiple strategies

### Advanced Features
- [ ] Intraday weather data (hourly updates)
- [ ] Extreme weather events (hurricanes, blizzards)
- [ ] Social media sentiment during weather events
- [ ] Energy futures as mediating variable

### Production Deployment
- [ ] Real-time weather API integration
- [ ] Live model retraining pipeline
- [ ] Risk management system (position sizing, stop-losses)
- [ ] Multi-asset portfolio optimization

---

## 📚 Dependencies

```
Core Libraries:
├─ pandas>=1.5.0
├─ numpy>=1.23.0
├─ scikit-learn>=1.2.0
├─ xgboost>=1.7.0
└─ matplotlib>=3.6.0

Analysis Tools:
├─ seaborn>=0.12.0
├─ shap>=0.41.0
├─ ta>=0.10.0 (Technical Analysis)
└─ backtesting>=0.3.3

Weather APIs:
├─ openmeteo-requests
├─ requests-cache
└─ retry-requests
```

---

## 📄 License

This project is for **educational and research purposes**. Stock data provided by Kaggle, weather data from NOAA/Open-Meteo under their respective terms of use.

---

## 🙏 Acknowledgments

- **Data Sources:** Kaggle (S&P 500), NOAA (Climate Data), Open-Meteo (API)
- **Course:** CX 4240 - Introduction to Computational Data Analysis @ Georgia Tech
- **Libraries:** Scikit-learn, XGBoost, TA-Lib, Backtesting.py

---

## 📧 Contact

**Elijah McCauley**  
Georgia Institute of Technology  
em5828@nyu.edu | [LinkedIn](https://linkedin.com/in/elijahmccauley) | [GitHub](https://github.com/elijahmccauley)

---
