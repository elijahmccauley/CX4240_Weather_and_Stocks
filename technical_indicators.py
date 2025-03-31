import pandas as pd
from ta import add_all_ta_features
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.utils import dropna

#load data
df = pd.read_csv('./stock_market_data/nyse/csv/AAC.csv', sep=',')
df = dropna(df)

# Exponential Moving Averages
ema15 = EMAIndicator(close=df["Close"], window=15).ema_indicator()
ema50 = EMAIndicator(close=df["Close"], window=50).ema_indicator()
df["ema15"] = ema15
df["ema50"] = ema50

# MACD
macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
df['macd'] = macd.macd()
df['macd_signal'] = macd.macd_signal()

# Bollinger Bands
bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
df['bb_bbm'] = bb.bollinger_mavg()
df['bb_bbh'] = bb.bollinger_hband()
df['bb_bbl'] = bb.bollinger_lband()

# RSI
rsi = RSIIndicator(close=df["Close"], window=14).rsi()
df["rsi"] = rsi

# Stochastic Oscillator
sto_osc = StochasticOscillator(high=df["High"], low=df["Low"], close=df["Close"], window=14, smooth_window=3)
df["stoch_k"] = sto_osc.stoch()
df["stoch_d"] = sto_osc.stoch_signal()

# Average True Range
atr = AverageTrueRange(high=df["High"], low=df["Low"], close=df["Close"], window=14).average_true_range()
df["atr_14"] = atr

# On Balance Volume
obv = OnBalanceVolumeIndicator(close=df["Close"], volume=df["Volume"]).on_balance_volume()
df["obv"] = obv

# Volume Weighted Average Price
vwap = VolumeWeightedAveragePrice(high=df["High"], low=df["Low"], close=df["Close"], volume=df["Volume"]).volume_weighted_average_price()
df["vwap"] = vwap

# Rate of Change
roc = ROCIndicator(close=df["Close"], window=10).roc()
df["roc_10"] = roc

# Increase
df["Increase"] = (df["Close"].shift(1) < df["Close"]).astype(int)


print(df.head())
df.to_csv("test_AAC.csv", index=False)
