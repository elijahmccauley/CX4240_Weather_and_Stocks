# CX4240_Weather_and_Stocks

Predicting stock prices with just indicators and comparing to predictions made with indicators and weather data.
https://www.kaggle.com/datasets/paultimothymooney/stock-market-data - stock data
https://www.ncei.noaa.gov/cdo-web/datatools
https://www.kaggle.com/datasets/nachiketkamod/weather-dataset-us - large weather
https://www.ncdc.noaa.gov/cdo-web/results - individual cities

To start, download the zip of the repository. This will include all the key requirements for replicating our results including: stock market data folder, standard stocks ipynb, and weather stock analysis ipynb.

Once this is downloaded, open up the standard_stoacks.ipynb notebook. In order to effectively recreate our results, you will want to run the cells in order, otherwise, you may run into some issues as dataframe and other variable names are resused multiple times. At the top, you will install the required libraries and import them. Then it will add the technical parameters to the base data set and make new csvs in a clean_data folder and then merge those into one large dataset called merged_data.csv. After that, your stock only dataset is ready to go and the rest of the cells should be fine to run in order without issue. In order to recreate the second half of the project, the weather data, you will first need to run the weather api notebook in order to get the weather data and merge it with the original dataset. Then you can run the weather_stock_analysis.ipynb file in order to replicate the weather based models, metrics, and visualizations.
