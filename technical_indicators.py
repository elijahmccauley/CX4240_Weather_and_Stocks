import pandas as pd
from ta import add_all_ta_features
from ta.utils import dropna

#load data
df = pd.read_csv('./stock_market_data/nyse/csv/AAC.csv', sep=',')
df = dropna(df)

df = add_all_ta_features(
    df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

print(df.head())
df["Target"] = (df["Close"].shift(1) < df["Close"]).astype(int)
print(df.head())
df.to_csv("test_AAC.csv", index=False)
