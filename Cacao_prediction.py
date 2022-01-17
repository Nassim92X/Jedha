from datetime import date
from fbprophet.plot import plot_plotly
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import itertools
import numpy as np
import random
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import plotly.io as pio
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
from matplotlib import pyplot

print("Data pour entrainer Prophet : de 1994 à 2021")
print("---------------------------------")
data = pd.read_csv(r"C:\Users\yahou\Downloads\Daily Prices_NEW.csv", delimiter=",", decimal=",", parse_dates=[0])
data["London futures (£ sterling/tonne)"] = data["London futures (£ sterling/tonne)"].str.replace(',', '')
data['London futures (£ sterling/tonne)'] = pd.to_numeric(data['London futures (£ sterling/tonne)'],errors = 'coerce')

print("Data pour analyse annuelle: de 2019 à 2021")
print("---------------------------------")
df = pd.read_csv(r"C:\Users\yahou\Downloads\Daily Prices_NEW (2).csv", delimiter=",", decimal=",", parse_dates=[0])
df["London futures (£ sterling/tonne)"] = data["London futures (£ sterling/tonne)"].apply(str).str.replace(',', '')
df['London futures (£ sterling/tonne)'] = pd.to_numeric(data['London futures (£ sterling/tonne)'],errors = 'coerce')
#print(data.info())


print("Exploration des données en vue journalière éclaté") 
print("---------------------------------") 
def date_features(df, label=None):
    df = df.copy()

    df['date'] = df.Date
    df['month'] = df['date'].dt.strftime('%B')
    df['year'] = df['date'].dt.strftime('%Y')
    df['dayofweek'] = df['date'].dt.strftime('%A')
    df['quarter'] = df['date'].dt.quarter
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X
X, y = date_features(df, label='London futures (£ sterling/tonne)')
df_new = pd.concat([X, y], axis=1)
print(df_new)

print("Statistique sur les prix des futurs")
print("---------------------------------")
print()
print(data['London futures (£ sterling/tonne)'].describe())


fig, ax = plt.subplots(figsize=(14,5))
palette = sns.color_palette("mako_r", 4)
a = sns.barplot(x="month", y="London futures (£ sterling/tonne)",hue = 'year',data=df_new)
a.set_title("Evolution Price Futur Contract on Cacao",fontsize=15)
plt.legend(loc='upper right')
plt.show()

data.plot(x='Date',y='London futures (£ sterling/tonne)',figsize=(15,6),linestyle='--', marker='*', markerfacecolor='r',color='y',markersize=10)
plt.xlabel('Years')
plt.ylabel('Price Cacao Futur')
plt.title("Evolution of the Cacao Stock Price")
plt.show()

#implémentation du modèle de ML 
print('implémentation du modèle de ML ')
print()
data = data[["Date","London futures (£ sterling/tonne)"]] # select Date and Price
data = data.rename(columns = {"Date":"ds","London futures (£ sterling/tonne)":"y"}) #renaming the columns of the dataset
data = data.reset_index(drop=True)
data = data[data['ds'].notna()]
#print(data)

from fbprophet import Prophet
m = Prophet(daily_seasonality = True) # the Prophet class (model)
m.fit(data) # fit the model using all data

future = m.make_future_dataframe(periods=365) #we need to specify the number of days in future
prediction = m.predict(future)
m.plot(prediction)
plt.title("Prediction of the Cacao Stock Price using the Prophet")
plt.xlabel("Date")
plt.ylabel("London futures Stock Price")
plt.show()

#Plot avec Matplotlib
m.plot_components(prediction)
plt.show()

#Plot avec Plotly
fig = plot_plotly(m, prediction)
fig.show()

