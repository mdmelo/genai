import os

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
# force use of tf_keras package (and not keras)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# avoid noisy log messages (this filters to Warning and above) - see https://github.com/tensorflow/tensorflow/issues/59779
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import tensorflow as tf
from tensorflow import keras


# define documents
docs = ['Well done!',
        'Good work',
        'Great effort',
        'nice work',
        'Excellent!',
        'Weak',
        'Poor effort!',
        'not good',
        'poor work',
        'Could have done better.']

# define class labels
labels = np.array([1,1,1,1,1,0,0,0,0,0])

# integer encode the documents.  keras one_hot() creates a hash of each
# word as an efficient integer encoding.
vocab_size = 50
encoded_docs = [keras.preprocessing.text.one_hot(d, vocab_size) for d in docs]

# [[33, 34], [38, 25], [14, 34], [2, 25], [43], [19], [9, 34], [41, 38], [9, 25], [1, 9, 34, 19]]
print(encoded_docs)


# pad documents to a max length of 4 words, makes all same length.
max_length = 4
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')
# [[33 34  0  0]
#  [38 25  0  0]
#  [14 34  0  0]
#  [ 2 25  0  0]
#  [43  0  0  0]
#  [19  0  0  0]
#  [ 9 34  0  0]
#  [41 38  0  0]
#  [ 9 25  0  0]
#  [ 1  9 34 19]]
print(padded_docs)

# define the model
model = keras.models.Sequential()

# Keras offers an Embedding layer that can be used for neural networks on text data.
#
# It requires that the input data be integer encoded, so that each word is represented
# by a unique integer. This data preparation step can be performed using the Tokenizer API
# also provided with Keras.
#
# The Embedding layer is initialized with random weights and will learn an embedding for all
# of the words in the training dataset.
#
# The Embedding layer is defined as the first hidden layer of a network. It must specify 3 arguments:
#
#     input_dim:    This is the size of the vocabulary in the text data.
#                   For example, if your data is integer encoded to values between 0-10,
#                   then the size of the vocabulary would be 11 words.
#
#     output_dim:   This is the size of the vector space in which words will be embedded.
#                   It defines the size of the output vectors from this layer for each word.
#                   For example, it could be 32 or 100 or even larger.
#
#     input_length: This is the length of input sequences, as you would define for any input
#                   layer of a Keras model. For example, if all of your input documents are
#                   comprised of 1000 words, this would be 1000.



# The output of the Embedding layer is a 2D vector with one embedding for each word in the input
# sequence of words (input document).

model.add(keras.layers.Embedding(vocab_size, 8, input_length=max_length))


# In order to connect a Dense layer directly to an Embedding layer, we must first flatten the
# 2D output matrix to a 1D vector using the Flatten layer.

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(1, activation='sigmoid'))


# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
# the output of the Embedding layer is a 4×8 matrix and this is squashed to a
# 32-element vector by the Flatten layer.
#
#   Model: "sequential"
#   _________________________________________________________________
#    Layer (type)                Output Shape              Param #
#   =================================================================
#    embedding (Embedding)       (None, 4, 8)              400
#
#    flatten (Flatten)           (None, 32)                0
#
#    dense (Dense)               (None, 1)                 33
#
#   =================================================================
#   Total params: 433 (1.69 KB)
#   Trainable params: 433 (1.69 KB)
#   Non-trainable params: 0 (0.00 Byte)
#   _________________________________________________________________
#   None
print(model.summary())


# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=1)


# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)

# varies 80% -> 100%
# 1/1 [==============================] - 2s 2s/step - loss: 0.6236 - accuracy: 1.0000
# Accuracy: 100.000000
print('Accuracy: %f' % (accuracy*100))
