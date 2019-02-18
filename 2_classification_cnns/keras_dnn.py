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
import tensorflow as tf

np.random.seed(RANDOM_SEED)

# Pull data from 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28*28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28*28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]

# =============================================================================
# Build model
# =============================================================================

print('\n-----------------------------------')
print('Build Neural Network')
print('-----------------------------------')

from keras import layers
from keras import models
from keras.regularizers import L1L2

model = models.Sequential()
model.add(layers.Dense(1000, activation='softmax', kernel_regularizer=L1L2(l1=0.0, l2=0.1),
                       input_dim=784))

model.summary()

print('\n-----------------------------------')
print(' Train the Model')
print('-----------------------------------')

model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5, batch_size=64)

print('End-to-end Runtime')
print('------------------------------------') 
end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time
print('\nThe average time end-to-end is',runtime,'seconds')

