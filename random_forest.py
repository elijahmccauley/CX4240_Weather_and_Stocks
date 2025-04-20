import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import export_text
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Best Parameters: {'max_depth': 20, 'max_features': None, 'min_samples_leaf': 4, 'min_samples_split': 10}

rf_classifier = RandomForestClassifier(
    n_estimators=200,         # number of trees in the forest
    max_depth=None,
    max_features='sqrt',      # good default for RF
    min_samples_leaf=1,
    min_samples_split=2,
)

rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy (RF): {test_accuracy}")

train_y_pred = rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, train_y_pred)
print(f"Train Accuracy (RF): {train_accuracy}")

importances = rf_classifier.feature_importances_
features = X.columns

plt.figure(figsize=(12, 6))
plt.barh(features, importances)
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importances')
plt.show()


# with 100, 20, sqrt, 4, 10
# Test Accuracy (RF): 0.5752187411797911
# Train Accuracy (RF): 0.8932648064641054

#After removing the days:

#Test Accuracy (RF): 0.5212768052300967
#Train Accuracy (RF): 0.7544869564836877


# 200, none, sqrt, 1, 2
#Test Accuracy (RF): 0.5783431463284957
#Train Accuracy (RF): 1.0