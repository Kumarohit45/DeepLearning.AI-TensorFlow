# Cats Vs Dogs
# Extracting the Zip File (The Cats and Dogs Dataset)
import zipfile

local_zip = './cats_and_dogs_filtered.zip'  # Mention the Directory Path
zip_ref = zipfile.ZipFile(local_zip, 'r')  # Read the Zip File
zip_ref.extractall()  # Extract all the contents of the Zip File

zip_ref.close()

import os

# Variable for Base Directory
base_dir = 'cats_and_dogs_filtered'

# Printing the contents of Base Directory
print('Contents of Base Directory : ')
print(os.listdir(base_dir))

# Printing the contents of Train Directory
print('\nContents of Train Directory : ')
print(os.listdir(f'{base_dir}/train'))

# Printing the contents of Validation Directory
print('\nContents of Validation Directory : ')
print(os.listdir(f'{base_dir}/validation'))

# Variable name to different directories
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')

validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

# Printing 10 names of Cats and Dogs present in Training Directory
print('\nPrinting 10 names of cats and dogs present in Training Directory :')
train_cats_fnames = os.listdir(train_cats_dir)
train_dogs_fnames = os.listdir(train_dogs_dir)

print(train_cats_fnames[:10])
print(train_dogs_fnames[:10])

# Printing 10 names of Cats and Dogs present in Validation Directory
validation_cats_fnames = os.listdir(validation_cats_dir)
validation_dogs_fnames = os.listdir(validation_dogs_dir)

print(validation_cats_fnames[:10])
print(validation_dogs_fnames[:10])

# Total number of Cat and Dog Images in Training and Validation Directory
print(f'\nTotal number of Cat Images in Training Directory : {len(os.listdir(train_cats_dir))}')
print(f'Total number of Dog Images in Training Directory : {len(os.listdir(train_dogs_dir))}')

print(f'Total number of Cat Images in Validation Directory : {len(os.listdir(validation_cats_dir))}')
print(f'Total number of Dog Images in Validation Directory : {len(os.listdir(validation_dogs_dir))}')

# Configure the matplotlib parameter
import matplotlib
import random

matplotlib.use('tkagg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

nrows = 4
ncols = 4

pic_index = 0

# Set up matplotlib fig and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index = pic_index + 8

next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cats_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dogs_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('off')  # Don't show axes (or gridlines)

    # img_path = random.choice(next_cat_pix + next_dog_pix)
    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# Build a Model
import tensorflow as tf

# Define a Callback
# class myCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         if logs is None:
#             logs = {}
#         if logs.get('accuracy') > 0.7200:
#             print('\nReached 72% Accuracy, So cancelling Training!!\n')
#             self.model.stop_training = True
#
#
# callbacks = myCallback()

model = tf.keras.models.Sequential([
    # Input Shape is desired size of the Image 150x150 with 3 bytes of color
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 Neuron Hidden Layers
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 Output Neuron, It will contain a value from 0-1 where 0 is for 'Cats Class' and 1 is for 'Dogs Class'
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Summary of the above Model
model.summary()

# Compile the Model
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Data Preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be scaled by 1.0/255.0
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Flow training images in batches of 20
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# Flow validation images in batches of 20
validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=20,
                                                        class_mode='binary',
                                                        target_size=(150, 150))

# Train the Model
history = model.fit(train_generator,
                    epochs=20,
                    validation_data=validation_generator,
                    verbose=2)

# Model Prediction
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

images = os.listdir('./cats')

print(images)

for i in images:
    print()
    path = './cats/' + i
    img = load_img(path, target_size=(150, 150))
    x = img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])
    if classes[0] > 0.5:
        print(i + ' is a dog.')
    else:
        print(i + ' is a cat.')

# Visualizing Intermediate Representations
successive_outputs = [layer.output for layer in model.layers]
visualization_model = tf.keras.models.Model(inputs=model.input, outputs=successive_outputs)

cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cats_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dogs_fnames]

img_path = random.choice(cat_img_files + dog_img_files)
img = load_img(img_path, target_size=(150, 150))
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

x = x / 255.0

successive_feature_maps = visualization_model.predict(x)

layer_names = [layer.name for layer in model.layers]

for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        n_features = feature_map.shape[-1]
        size = feature_map.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            np.seterr(invalid='ignore')
            x = feature_map[0, :, :, i]
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size: (i + 1) * size] = x

        scale = 20. / n_features
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()

# Evaluating Accuracy and Loss for the Model
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and Validation Loss')
plt.show()
