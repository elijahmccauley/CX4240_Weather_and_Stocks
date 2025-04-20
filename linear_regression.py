import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

"""stock_folder = "./clean_data"
stock_files = [f for f in os.listdir(stock_folder) if f.endswith('.csv')]
print(stock_files)
dataframes = []

for file in stock_files:
    file_path = os.path.join(stock_folder, file)
    print(file, file_path)
    df = pd.read_csv(file_path)
    df['Ticker'] = file.split('.')[0]
    dataframes.append(df)

merged_df = pd.concat(dataframes, ignore_index=True)
merged_df.to_csv('merged_data.csv', index=False)"""

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

"""param_grid = {
    'max_depth': [3, 5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)"""
# Best Parameters: {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 10}


reg = LinearRegression()
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse}")

threshold = 0.005  # 0.5%

profitable_predictions = (y_pred > threshold)
num_trades = profitable_predictions.sum()
avg_actual_return = y_test[profitable_predictions].mean()

print(f"Number of trades: {num_trades}")
print(f"Avg return on selected trades: {avg_actual_return:.4f}")

# base model avg returns on selected trades = 1.32%