# =============================================================================
# RNN -- Reuters Dataset -- HW 3 
# =============================================================================
# Code created by paulhuynh
# Adapted by John Kiley
# Code creation date: 2018.11.10
# Code modification date: 2018.11.16
# =============================================================================
# Load Packages
# =============================================================================

from keras.models import Sequential
from keras import layers
from keras.datasets import reuters

from sklearn.metrics import confusion_matrix, accuracy_score, auc

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

start_time = time.clock()
# =============================================================================
# Load in Reuters data 
# =============================================================================

# Load data into splits
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words = 10000)

#word index if desired to understand what each number represents
word_index = reuters.get_word_index(path="reuters_word_index.json")

#function used to transform the data into indivual documents to be fed into model
def vec_seq(seqs, dim = 10000):
    results = np.zeros((len(seqs), dim))
    for i, seq in enumerate(seqs):
        results[i,seq] = 1
    return results


#one hot encoding labels
def one_hot(labels, dim = 46):
    results = np.zeros((len(labels),dim))
    for i, label in enumerate(labels):
        results[i,label] = 1
    return results
    
#creating datasets to be used for model
x_train_seq = vec_seq(x_train)
x_test_seq = vec_seq(x_test)

y_train_one_hot = one_hot(y_train)
y_test_one_hot = one_hot(y_test)

# =============================================================================
# Define the model
# =============================================================================
# This model has dense layers only, fully connected model.

#function to create model
def model():
    model = Sequential()
    model.add(layers.Embedding(10000,32))
    model.add(layers.SimpleRNN(46))
    model.add(layers.Dense(46,activation='softmax'))
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

#binary classification uses softmax and categorical_crossentropy 
print('\n----------------------------------------')
print('\nTrain the model')
print('\n----------------------------------------')

model = model()
history = model.fit(x_train_seq,
                    y_train_one_hot,
                    epochs = 5,    
                    batch_size = 32,
                    validation_data=(x_test_seq,y_test_one_hot),                    
                    verbose = True)

# =============================================================================
# Plotting the training and testing loss
# =============================================================================
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss,'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Describe the model build
print('\n----------------------------------------')
print('\nSummary of the model')
print('\n----------------------------------------')


model.summary()

# predictions
pred = model.predict_classes(x_test_seq)

# =============================================================================
# Model Evaluation
# =============================================================================

#Confusion Matrix
matrix = pd.DataFrame(confusion_matrix(y_test,pred, labels = [x for x in range(0,46)]))

# Accuracy
print('\n----------------------------------------')
score = accuracy_score(y_test,pred)
print('\nThe accuracy score is',score)

print('\n----------------------------------------')
# Model running performance
print('\nEnd-to-end Runtime')
end_time = time.clock()
runtime = end_time - start_time  # seconds of wall-clock time
print('\nThe average time end-to-end is',runtime,'seconds')
print('\n----------------------------------------')
