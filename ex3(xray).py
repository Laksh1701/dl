import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from tensorflow.keras.models import Model, Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow import keras

img_width, img_height = 64, 64
batchsize = 64
num_of_class = 2

train = keras.utils.image_dataset_from_directory(
directory='/Users/spikezaidspeigel/Desktop/dl lab/chest_xray/chest_xray/train',
labels='inferred',
label_mode='categorical',
batch_size=batchsize,
image_size=(img_width, img_height))
test = keras. utils.image_dataset_from_directory(
directory='/Users/spikezaidspeigel/Desktop/dl lab/chest_xray/chest_xray/test',
labels='inferred',
label_mode='categorical',
batch_size=batchsize,
image_size=(img_width, img_height))

x_train = []
y_train = []
x_test = []
y_test = []
for feature, label in train:
    x_train.append(feature.numpy())
    y_train.append(label.numpy())
for feature, label in test:
    x_test.append(feature.numpy())
    y_test.append(label.numpy())
x_train = np.concatenate(x_train, axis=0)
x_test = np.concatenate(x_test, axis=0)
y_train = np.concatenate(y_train, axis=0)
y_test = np.concatenate(y_test, axis=0)
x_train=x_train/256
x_test=x_test/256
print(x_train.shape)
print(x_test.shape)


input_shape = img_width, img_height, 3
input_img = Input(shape=input_shape)
x = Conv2D(32, (8, 8), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (8, 8), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (8, 8), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (8, 8), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (8, 8), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (8, 8), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (8, 8), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train, x_train, epochs=5,batch_size=64, shuffle=True, validation_data=