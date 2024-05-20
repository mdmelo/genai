import os

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
# force use of tf_keras package (and not keras)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# avoid noisy log messages (this filters to Warning and above) - see https://github.com/tensorflow/tensorflow/issues/59779
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# The Keras Embedding layer can also use a word embedding learned elsewhere.
# It is common in NLP to learn, save, and make freely available word embeddings.
# see https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.py

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


# prepare tokenizer

# Tokenizer class can be fit on the training data, convert text
# to sequences consistently by calling the texts_to_sequences()
# method on the Tokenizer class, and provides access to the dictionary
# mapping of words to integers in a word_index attribute
t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(docs)

vocab_size = len(t.word_index) + 1

# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)

# [[6, 2], [3, 1], [7, 4], [8, 1], [9], [10], [5, 4], [11, 3], [5, 1], [12, 13, 2, 14]]
print(encoded_docs)

# pad documents to a max length of 4 words
max_length = 4
padded_docs = keras.preprocessing.sequence.pad_sequences(encoded_docs, maxlen=max_length, padding='post')

# [[ 6  2  0  0]
#  [ 3  1  0  0]
#  [ 7  4  0  0]
#  [ 8  1  0  0]
#  [ 9  0  0  0]
#  [10  0  0  0]
#  [ 5  4  0  0]
#  [11  3  0  0]
#  [ 5  1  0  0]
#  [12 13  2 14]]
print(padded_docs)

# load the whole embedding into memory
embeddings_index = dict()

# glove.6B.100 package (from Stanford) of embeddings is 822Mb. It was trained on
# a dataset of one billion tokens (words) with a vocabulary of 400 thousand words.
# There are a few  different embedding vector sizes, including 50, 100, 200 and
# 300 dimensions.
#
# Each entry in this file is a token (word) followed by its weights (100 numbers).
f = open('/books/MachineLearning/GenerativeDeepLearning/data/glove.6B.100d.txt')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()

# Loaded 400000 word vectors.
print('Loaded %s word vectors.' % len(embeddings_index))

# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in t.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Embedding matrix, training docs using glove.6B embeddings
#  [[ 0.          0.          0.         ...  0.          0.           0.        ]
#   [-0.11619     0.45447001 -0.69216001 ... -0.54737002  0.48822001   0.32246   ]
#   [-0.2978      0.31147    -0.14937    ... -0.22709    -0.029261     0.4585    ]
#  ...
#   [ 0.05869     0.40272999  0.38633999 ... -0.35973999  0.43718001   0.10121   ]
#   [ 0.15711001  0.65605998  0.0021149  ... -0.60614997  0.71004999   0.41468999]
#   [-0.047543    0.51914001  0.34283999 ... -0.26859     0.48664999   0.55609   ]]
print("Embedding matrix, training docs using glove.6B embeddings\n", embedding_matrix)

# define model
model = keras.models.Sequential()
e = keras.layers.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# summarize the model
#
#     Model: "sequential"
#     _________________________________________________________________
#      Layer (type)                Output Shape              Param #
#     =================================================================
#      embedding (Embedding)       (None, 4, 100)            1500
#
#      flatten (Flatten)           (None, 400)               0
#
#      dense (Dense)               (None, 1)                 401
#
#     =================================================================
#     Total params: 1901 (7.43 KB)
#     Trainable params: 401 (1.57 KB)
#     Non-trainable params: 1500 (5.86 KB)
#     _________________________________________________________________
print(model.summary())

# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=1)

# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=1)

# 1/1 [==============================] - 2s 2s/step - loss: 0.2709 - accuracy: 1.0000
# Accuracy: 100.000000
print('Accuracy: %f' % (accuracy*100))