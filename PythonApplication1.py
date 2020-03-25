import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

import sys

# GPU fix for tensorflow/cuDNN
import tensorflow
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# input image dimensions
img_rows, img_cols = 28, 28
num_classes = 10
batch_size = 128

# the data, split between train and test sets
(images_tr, labels_tr), (images_test, labels_test) = mnist.load_data()

# Reshape data for the network: number of items, c h w/h w c depending on the keras image data format
if K.image_data_format() == 'channels_first':
    x_train = images_tr.reshape(images_tr.shape[0], 1, img_rows, img_cols)
    x_test = images_test.reshape(images_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = images_tr.reshape(images_tr.shape[0], img_rows, img_cols, 1)
    x_test = images_test.reshape(images_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalize
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(labels_tr, num_classes)
y_test = keras.utils.to_categorical(labels_test, num_classes)
# Print sets
print("Train set", x_train.shape, y_train.shape)
print("Test set", x_test.shape, y_test.shape)

# Data is fine, configure simple sequential model. TODO: try make more poolings
cnn_model = Sequential()
cnn_model.add(Conv2D(128, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(64, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Conv2D(32, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
cnn_model.add(Flatten())
# cnn_model.add(Dropout(0.5))
# cnn_model.add(Dense(128))
cnn_model.add(Dense(num_classes, activation='softmax'))
cnn_model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy']
)
cnn_model.summary()
ld_st_action = input("Load weights? ")

if ld_st_action == "Y" or ld_st_action == "y":
    cnn_model.load_weights("mnist_model.h5")
else:
    print("Start training\n") 
    cnn_model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=10,
        verbose=1,
        validation_data=(x_test, y_test))
    print("Save model\n") 
    cnn_model.save_weights("mnist_model.h5")

score = cnn_model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Render 1st image with label
fig = plt.figure()
for i in range(9):
  plt.subplot(3,3,i+1)
  plt.tight_layout()
  plt.imshow(images_test[i], cmap='gray', interpolation='none')
  predicted = cnn_model.predict(np.array([x_test[i]]))[0]
  plt.title("Test: {real} predicted {cnnval}".format(real=labels_test[i], cnnval=np.argmax(predicted)))
  plt.xticks([])
  plt.yticks([])
fig
plt.show()

