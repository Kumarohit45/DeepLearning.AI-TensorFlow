# Neural Network Method
# Importing the libraries and modules
import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)
print(keras.__version__)

# Build a simple sequential model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Declare model inputs and outputs for training
xs = np.array([-7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=float)
ys = np.array([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=500)

# Make a prediction
print("Using Neural Network Model : ", model.predict([22.0]))


# Primitive Method
def hw_function(x):
    y = (x * 2) - 1
    return y


ans = hw_function(22)
print("Using Primitive Method : ", ans)
