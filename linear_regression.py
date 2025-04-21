import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV


stock_data = pd.read_csv('./merged_data.csv')
stock_data['Pct_Change'] = (stock_data['Close'].shift(-1) - stock_data['Close']) / stock_data['Close']
stock_data = stock_data.dropna()
#label_encoder = LabelEncoder()
#stock_data['Ticker'] = label_encoder.fit_transform(stock_data['Ticker'])
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data['Year'] = stock_data['Date'].dt.year
stock_data['Month'] = stock_data['Date'].dt.month
stock_data['Day'] = stock_data['Date'].dt.day
stock_data = stock_data.drop('Date', axis=1)
stock_data = stock_data.drop('Symbol', axis=1)
stock_data = stock_data.drop('Ticker', axis=1)

X = stock_data.drop(['Pct_Change'], axis=1)
y = stock_data['Pct_Change']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

threshold = 0.005  # 0.5%

profitable_predictions = (y_pred > threshold)
num_trades = profitable_predictions.sum()
returns = y_test[profitable_predictions]
quants = np.quantile(returns, [0, 0.05, 0.5, .95, 1])
normalized_returns = []
for i in returns:
    if i > quants[1] and i < quants[3]:
        normalized_returns.append(i)

avg_actual_return = y_test[profitable_predictions].mean()
avg_selected_returns = sum(normalized_returns) / len(normalized_returns)

print(f"Number of selected trades: {num_trades}")
print(f"Number of middle 90% trades: {len(normalized_returns)}")
print(f"Avg return on all selected trades: {avg_actual_return:.4f}")
print(f"Avg return on filtered selected trades: {avg_selected_returns:.4f}")
print(y_test[profitable_predictions].max())
# base model avg returns on selected trades = 1.32%
# returns of middle 90% = 0.0113