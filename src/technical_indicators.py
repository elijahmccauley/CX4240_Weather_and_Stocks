import pandas as pd
from ta import add_all_ta_features
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.utils import dropna
import numpy as np

#load data
df = pd.read_csv('./archive/sp500_stocks.csv', sep=',')
df = dropna(df)

company_df = pd.read_csv('./archive/sp500_companies.csv', sep=',')
companies = company_df["Symbol"]
#print(companies)
companies = companies.tolist
#print(companies)

company_split = {company: df[df["Symbol"] == company] for company in df["Symbol"].unique()}
for group, sub_df in company_split.items():
    
    print(f"Group {group}:\n{sub_df}\n")

    # Exponential Moving Averages
    ema15 = EMAIndicator(close=sub_df["Close"], window=15).ema_indicator()
    ema50 = EMAIndicator(close=sub_df["Close"], window=50).ema_indicator()
    sub_df["ema15"] = ema15
    sub_df["ema50"] = ema50

    # MACD
    macd = MACD(close=sub_df["Close"], window_slow=26, window_fast=12, window_sign=9)
    sub_df['macd'] = macd.macd()
    sub_df['macd_signal'] = macd.macd_signal()

    # Bollinger Bands
    bb = BollingerBands(close=sub_df["Close"], window=20, window_dev=2)
    sub_df['bb_bbm'] = bb.bollinger_mavg()
    sub_df['bb_bbh'] = bb.bollinger_hband()
    sub_df['bb_bbl'] = bb.bollinger_lband()

    # RSI
    rsi = RSIIndicator(close=sub_df["Close"], window=14).rsi()
    sub_df["rsi"] = rsi

    # Stochastic Oscillator
    sto_osc = StochasticOscillator(high=sub_df["High"], low=sub_df["Low"], close=sub_df["Close"], window=14, smooth_window=3)
    sub_df["stoch_k"] = sto_osc.stoch()
    sub_df["stoch_d"] = sto_osc.stoch_signal()

    # Average True Range
    atr = AverageTrueRange(high=sub_df["High"], low=sub_df["Low"], close=sub_df["Close"], window=14).average_true_range()
    sub_df["atr_14"] = atr

    # On Balance Volume
    obv = OnBalanceVolumeIndicator(close=sub_df["Close"], volume=sub_df["Volume"]).on_balance_volume()
    sub_df["obv"] = obv

    # Volume Weighted Average Price
    vwap = VolumeWeightedAveragePrice(high=sub_df["High"], low=sub_df["Low"], close=sub_df["Close"], volume=sub_df["Volume"]).volume_weighted_average_price()
    sub_df["vwap"] = vwap

    # Rate of Change
    roc = ROCIndicator(close=sub_df["Close"], window=10).roc()
    sub_df["roc_10"] = roc

    # Increase
    sub_df["Target"] = (sub_df["Close"].shift(-1) > sub_df["Close"]).astype(int)

    # Clopen
    sub_df["Clopen"] = sub_df["Close"]/sub_df["Open"]

    # High/Low
    sub_df["HighLow"] = sub_df["High"]/sub_df["Low"]

    # Log Change
    sub_df["log_price"] = np.log(sub_df["Close"])
    sub_df["Log5"] = sub_df["log_price"] - sub_df["log_price"].shift(5)
    sub_df["Log15"] = sub_df["log_price"] - sub_df["log_price"].shift(15)
    sub_df["Log30"] = sub_df["log_price"] - sub_df["log_price"].shift(30)

    print(sub_df.head(), group)
    sub_df.to_csv("{}.csv".format(group), index=False)
