import numpy as np
import pandas as pd
dataset = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv").astype('float32')
dataset.rename(columns={'0':'label'}, inplace=True)
x= dataset.drop('label',axis = 1)
y = dataset['label']

print(x.shape)
print(y.shape)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
x_train,x_test,y_train,y_test = train_test_split(x,y)
standard_scaler = MinMaxScaler()
standard_scaler.fit(x_train)
x_train = standard_scaler.transform(x_train)
x_test = standard_scaler.transform(x_test)
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
print(x_train.shape)

print(x_test.shape)
print(y_train.shape)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
x_train.shape[1]
model = keras.Sequential()
model.add(layers.Dense(300, activation="relu" , input_dim = x_train.shape[1]))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(200, activation="relu"))
model.add(layers.Dense(300, activation="relu"))
model.add(layers.Dense(100, activation="relu"))
model.add(layers.Dense(len(y.unique()), activation="softmax"))
adam = keras.optimizers.Adam(lr=0.001,decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam , metrics = ['Accuracy'])
model.fit(x_train,y_train,epochs=100)
model.summary()
model.evaluate(x_test,y_test)
import matplotlib.pyplot as plt
plt.imshow(x_test[100].reshape(28,28), cmap='Greys')
plt.show()
k=model.predict(x_test[250].reshape(1,784))
np.argmax(k)
