import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import export_text
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree

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

X = stock_data.drop('Target', axis=1)
y = stock_data['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

plot_tree(dt_classifier)

