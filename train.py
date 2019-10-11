
# These lines import the necessary libraries (collections of functions) for the program
# from ... import ... => simply imports specific functions from the libraries
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import tensorflow as tf
import IPython.display as display
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE

# This pathlib.Path() is a better way of handling file paths rather than saving hardcoded string paths.
# This works on Macs, Linux, and Windows
train_dir = pathlib.Path('train')
test_dir = pathlib.Path('test')

# This finds the class folders under the train data folder to save as the class names. works for any number of classes
CLASS_NAMES = np.array([item.name for item in train_dir.glob('*') if item.name != "train.py"])
print(CLASS_NAMES)

# Calculates the number of images in the test/train and attentive/distracted folders
num_att_train = len(os.listdir(train_dir / 'attentive'))
num_dis_train = len(os.listdir(train_dir / 'distracted'))
num_att_test = len(os.listdir(test_dir / 'attentive'))
num_dis_test = len(os.listdir(test_dir / 'distracted'))

total_train = num_att_train + num_dis_train
total_test = num_att_test + num_dis_test

print('Train Attentive:  ' + str(num_att_train))
print('Train Distracted: ' + str(num_dis_train))
print('Test Attentive:   ' + str(num_att_test))
print('Test Distracted:  ' + str(num_dis_test))

# Creates image preprocessors on the data. Look this up by googling "keras imagedatagenerator"
train_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, 
                                                                        rotation_range=15,
                                                                        zoom_range=0.10)
test_image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Establishes variables for training
BATCH_SIZE = 32
IMG_HEIGHT = 300
IMG_WIDTH = 300
EPOCHS = 30

# This flow from directory creates little batches of images that are randomized using the image data generators
# from above.
train_data_gen = train_image_generator.flow_from_directory(directory=str(train_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     classes = list(CLASS_NAMES))

test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=str(test_dir),
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH))

# Displays some example images
def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(BATCH_SIZE):
        ax = plt.subplot(1,4,n+1)
        plt.imshow(image_batch[n])
        plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
        plt.axis('off')
    plt.show()
        
# Shows an example batch. Next() just generates another batch of images from the train_data_gen
image_batch, label_batch = next(train_data_gen)
#show_batch(image_batch, label_batch)

print(np.shape(image_batch))
print(np.shape(label_batch))

# Defines a simple convolutional neural network. Try changing parameters here.
# Also notice that the images are grayscale (2D array) but we convert them to RGB images with 3 channels. This is not necessary. Try taking a grayscale input
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    #Dropout(0.2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='sigmoid')
])

# Because we will almost never have equal number of attentive and distracted images, we have to weight the classes to correct for the imbalance.
# For a better understanding of this, google 'machine learning class imbalance class weighting'
# These lines create a list to be used as class weights for training.
class_one_items = len(os.listdir(train_dir / CLASS_NAMES[0])) # How many images are the in the first class
class_weights = [class_one_items/len(os.listdir(train_dir / CLASS)) for CLASS in CLASS_NAMES] # Creates weights so that the product of the number of images in the class 

print("Class Weights", class_weights) #prints weights for user verification


# This defines some additional parameters for training.
# Loss is what's minimized when training
# Accuracy is what we monitor to determine overfitting and final performance
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# This is where training happens. History is simply the variable that stores the output from the training session.
history = model.fit_generator(
    train_data_gen, # where the model gets the training data from
    steps_per_epoch=total_train // BATCH_SIZE, # number of batches per epoch. An epoch is a complete training cycle through the entire training set
    epochs=EPOCHS,
    validation_data=test_data_gen, # where it gets the validation data from
    validation_steps=total_test // BATCH_SIZE, # number of batches per epoch for validation
    class_weight=[1,3]) # used to equalize the differences between the number of attentive picutres we have and the number of distracted pictures
                        # Since we have so many more attentive pictures than distracted ones, we have to weight the distracted pictures more (9x in this case)
                        # This increases the loss by 9 times when the model is training on a distracted picture

# Saves the trained model
model.save('trained_hand_model.h5')

# Gets the training and validation accuracy and loss of the training session
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Displays the training graphs
epochs_range = range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


