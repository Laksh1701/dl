import numpy as np
import tensorflow as tf
X = np.array([[0, 0],
[0, 1],
[1, 0],
[1, 1]])
y = np.array([[0],
[1],
[1],
[0]])
model = tf.keras.Sequential([
tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,), name='hidden_layer'),
tf.keras.layers.Dense(1, activation='sigmoid', name='output_layer')
])
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
history = model.fit(X, y, epochs=3000)
predictions = model.predict(X)
for i in range(len(X)):
    print(f"{X[i]} XOR =", round(predictions[i][0]))
