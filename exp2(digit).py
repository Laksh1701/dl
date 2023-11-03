import tensorflow as tf
import tensorflow_datasets
import matplotlib.pyplot as plt
import numpy as np
(X_train,y_train),(X_test,y_test)=tf.keras.datasets.mnist.load_data()

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation="relu"))
model.add(tf.keras.layers.Dense(10,activation="softmax"))
model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(X_train,y_train, epochs=5)

pred=model.predict(X_test)
plt.imshow(X_test[0])
print(np.argmax(pred[0]))
