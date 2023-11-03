import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_excel("/content/Google Dataset.xlsx")
data[data['Change %']==0.0]
data['Month Starting'] = pd.to_datetime(data['Month Starting'], errors='coerce').dt.date

data['Month Starting'][31] = pd.to_datetime('2020-05-01')
data['Month Starting'][43] = pd.to_datetime('2019-05-01')
data['Month Starting'][55] = pd.to_datetime('2018-05-01')

plt.figure(figsize=(25,5))
plt.plot(data['Month Starting'],data['Open'], label='Open')
plt.plot(data['Month Starting'],data['Close'], label='Close')
plt.xlabel('Year')
plt.ylabel('Close Price')
plt.legend()
plt.title('Change in the stock price of Google over the years')

# Calculating the daily returns
data['Returns'] = data['Close'].pct_change()
# Calculating the rolling average of the returns
data['Rolling Average'] = data['Returns'].rolling(window=30).mean()
plt.figure(figsize=(10,5))
''' Creating a line plot using the 'Month Starting' column as the x-axis
and the 'Rolling Average' column as the y-axis'''
sns.lineplot(x='Month Starting', y='Rolling Average', data=data)

corr = data.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot=True, cmap='coolwarm')

scaler = StandardScaler()
data['Returns'] = scaler.fit_transform(data['Returns'].values.reshape(-1,1))
data.head()

data['Returns'] = data['Returns'].fillna(data['Returns'].mean())
data['Rolling Average'] = data['Rolling Average'].fillna(data['Rolling Average'].mean())

from sklearn.ensemble import IsolationForest
model = IsolationForest(contamination=0.05)
model.fit(data[['Returns']])
# Predicting anomalies
data['Anomaly'] = model.predict(data[['Returns']])
data['Anomaly'] = data['Anomaly'].map({1: 0, -1: 1})
# Ploting the results
plt.figure(figsize=(13,5))
plt.plot(data.index, data['Returns'], label='Returns')
plt.scatter(data[data['Anomaly'] == 1].index, data[data['Anomaly'] == 1]['Returns'], color='red')
plt.legend(['Returns', 'Anomaly'])
plt.show()