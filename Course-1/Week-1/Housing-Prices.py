# A house has a base cost of 50k, and every additional bedroom adds a cost of 50k. This will make a 1-bedroom house
# cost 100k, a 2-bedroom house cost 150k etc. How would you create a neural network that learns this relationship so
# that it would predict a 7-bedroom house as costing close to 400k etc.

# Importing the libraries and modules
import tensorflow as tf
import numpy as np
from tensorflow import keras


# Creating a house_model function
def house_model():
    # Define input and output tensors with the value of houses with 1 upto 6 bedrooms
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

    # Define the model (neuron), model with 1 dense layer and 1 unit
    neuron = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

    # Compile the model using optimizer as Stochastic Gradient Descent and Loss as Mean Squared Error
    neuron.compile(optimizer='sgd', loss='mean_squared_error')

    # Training model for 1000 epochs by feeding the I/O tensors
    neuron.fit(xs, ys, epochs=1000)

    # Returning the model
    return neuron


# Calling the house_model function
model = house_model()

# Make a prediction
new_x = 7.0
prediction = model.predict([new_x])[0]
print(prediction)
