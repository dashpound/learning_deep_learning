# =============================================================================
# Setup work environment
# =============================================================================

# ensure common functions across Python 2 and 3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import time

start_time = time.clock()

RANDOM_SEED = 9999  # to ensure reproducibility of study results

# standard normal random number generator used to generate data and noise
np.random.seed(RANDOM_SEED)

#%%
# =============================================================================
# Import data/Generate data
# =============================================================================
print('\n-----------------------------------')
print('Import Data for Neural Network')
print('-----------------------------------')

from keras.datasets import mnist
from keras.utils import to_categorical

# Pull data from 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000,28,28,1)).astype('float32')/255
X_test = X_test.reshape((10000,28,28,1)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# =============================================================================
# Build model
# =============================================================================

print('\n-----------------------------------')
print('Build Neural Network')
print('-----------------------------------')

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2,)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

print('\n-----------------------------------')
print(' Train the Model')
print('-----------------------------------')

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64)

print('End-to-end Runtime')
print('------------------------------------') 
end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time
print('\nThe average time end-to-end is',runtime,'seconds')

