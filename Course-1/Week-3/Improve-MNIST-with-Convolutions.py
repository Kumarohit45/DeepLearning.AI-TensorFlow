import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

mnist = tf.keras.datasets.mnist
(training_images, training_labels) = mnist.load_data()
print(type(training_images))
timages = np.asarray(training_images)
print(type(timages))
timages = timages.reshape(60000, 28, 28, 1)

timages = timages / 255.0

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.9950):
      print("Reached 99.5% accuracy so cancelling training!")
      self.model.stop_training = True


callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(timages, training_labels, epochs=20, callbacks=[callbacks])
