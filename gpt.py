#!/usr/bin/env python
# coding: utf-8

# see https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
#     https://jalammar.github.io/illustrated-transformer/

# installed packages
#   tensorboard                  2.16.2
#   tensorboard-data-server      0.7.2
#   tensorboard-plugin-wit       1.8.1
#   tensorflow                   2.16.1
#   tensorflow-cpu               2.16.1
#   tensorflow-estimator         2.10.0
#   tensorflow-io-gcs-filesystem 0.36.0
#   tensorflow-probability       0.18.0
#   tf_keras                     2.16.0


import os

# using TF2.x with Keras 2.x see https://keras.io/getting_started/ and https://github.com/tensorflow/tensorflow/issues/63849
# force use of tf_keras package (and not keras)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# avoid noisy log messages (this filters to Warning and above) - see https://github.com/tensorflow/tensorflow/issues/59779
# ex 'Could not load dynamic library 'libnvinfer.so.7' which is only needed for nvidia TensorRT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import json
import re
import string

import tensorflow as tf
from tensorflow import keras
from IPython.display import HTML


LOAD_MODEL = True

VOCAB_SIZE = 10000
MAX_LEN = 80
EMBEDDING_DIM = 256
KEY_DIM = 256
N_HEADS = 2
FEED_FORWARD_DIM = 256
VALIDATION_SPLIT = 0.2
SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
SAMPLEINP = 25



with open("/books/MachineLearning/GenerativeDeepLearning/data/winemag-data-130k-v2.json") as json_data:
    wine_data = json.load(json_data)


# Filter the dataset
filtered_data = ["wine review : " + x["country"] + " : " + x["province"] + " : " + x["variety"] + " : " + x["description"]
    for x in wine_data
    if x["country"] is not None
    and x["province"] is not None
    and x["variety"] is not None
    and x["description"] is not None
]

n_wines = len(filtered_data)
print(f"{n_wines} recipes loaded")

example = filtered_data[SAMPLEINP]
print("sample filtered data: ", example)

# Preparing the data:
#   1. Load the data and create a list of text string descriptions of each wine.
#
#   2. Pad punctuation with spaces, so that each punctuation mark is treated as a separate word.
#
#   3. Pass the strings through a TextVectorization layer that tokenizes the data and pads/clips
#      each string to a fixed length.
#
#   4. Create a training set where the inputs are the tokenized text strings and the outputs
#      to predict (eg labels) are the same strings shifted by one token (for next-word prediction).



# -- Tokenize the data

# pad the punctuation with spaces - this treats punctuation as separate words
def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}, '\n'])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


text_data = [pad_punctuation(x) for x in filtered_data]

# display an example of a recipe
example_data = text_data[SAMPLEINP]
print("sample with punctuation adjust: ", example_data)

# convert to a Tensorflow Dataset
#
# The given tensor (a list of strings) is sliced along its first dimension. 
# This operation preserves the structure of the input tensor but removes 
# the first dimension, using it as the dataset dimension. 
#
# text_ds = 129_907 documents (each max 80 characters), in 32 document batches
# which makes for 4060 training passes per epoch (129_907 / 32 = 4059.59)
# ... 4060/4060 [==============================] - ETA: 0s - loss: 1.9597 - dense_2_loss: 1.9597
text_ds = tf.data.Dataset.from_tensor_slices(text_data).batch(BATCH_SIZE).shuffle(1000)



# Create a text vectorization layer
#
# TextVectorization is a preprocessing layer which maps text features to integer sequences.
# It transforms a batch of strings (one example = one string) into either a list of token indices
# (one example = 1D tensor of integer token indices) or a dense representation (one example =
# 1D tensor of float values representing data about the example's tokens).
#
# Since a vocabulary is not supplied, this layer learns the vocabulary. It will analyze
# the dataset, determine the frequency of individual string values, and create the  vocabulary.
#
# due to TF lazy loading alot of setup code runs before the TV code, eventually execution
# lands at site-packages/tf_keras/src/layers/preprocessing/text_vectorization.py (~ line 266) -
# which executes base class site-packages/tf_keras/src/engine/base_layer.py (~ line 741) to create
# the layer.
#
# see https://keras.io/api/layers/preprocessing_layers/text/text_vectorization/ 
vectorize_layer = keras.layers.TextVectorization(standardize="lower",
                                                 max_tokens=VOCAB_SIZE,
                                                 output_mode="int",
                                                 output_sequence_length=MAX_LEN + 1)



# Adapt the layer to the training set
#
# Calling `adapt()` on a `TextVectorization` layer is an alternative to
# passing in a precomputed vocabulary on construction via the `vocabulary`
# argument. A `TextVectorization` layer should always be either adapted
# over a dataset or supplied with a vocabulary.
# 
# During `adapt()`, the layer will build a vocabulary of all string tokens
# seen in the dataset, sorted by occurrence count, with ties broken by
# sort order of the tokens (high to low). At the end of `adapt()`, if
# `max_tokens` is set, the vocabulary wil be truncated to `max_tokens`
# size. For example, adapting a layer with `max_tokens=1000` will compute
# the 1000 most frequent tokens occurring in the input dataset. If
# `output_mode='tf-idf'`, `adapt()` will also learn the document
# frequencies of each token in the input dataset.
# 
# In order to make `TextVectorization` efficient in any distribution
# context, the vocabulary is kept static with respect to any compiled
# `tf.Graph`s that call the layer. As a consequence, if the layer is
# adapted a second time, any models using the layer should be re-compiled.
# 
# `adapt()` is meant only as a single machine utility to compute layer
# state.  To analyze a dataset that cannot fit on a single machine, see
# https://www.tensorflow.org/tfx/transform/get_started) for a multi-machine, 
# map-reduce solution.
vectorize_layer.adapt(text_ds)
print("token create from vocabulary complete")

# see keras/src/layers/preprocessing/text_vectorization.py
#     keras/src/layers/preprocessing/index_lookup.py
vocab = vectorize_layer.get_vocabulary()

# Display the same example converted to ints
example_tokenised = vectorize_layer(example_data)
print("sample converted to vocabulary tokens: ", example_tokenised.numpy())


# Display some token:word mappings
print("some token -> word mappings")
for i, word in enumerate(vocab[::1000]):
    print(f"{i * 1000}: {word}")




# -- Create the Training Set


# create the same text shifted by one word, will be added to training set
def prepare_inputs(text):
    # `axis=-1` adds an inner most dimension
    text = tf.expand_dims(text, -1)
    tokenized_sentences = vectorize_layer(text)
    x = tokenized_sentences[:, :-1]
    y = tokenized_sentences[:, 1:]
    print("returning {} and {}".format(x,y))
    return x, y

# creates the training set of recipes using the function directly above.
#
# This transformation applies `map_func` to each element of this dataset, and
# returns a new dataset containing the transformed elements, in the same
# order as they appeared in the input. `map_func` can be used to change both
# the values and the structure of a dataset's elements. Supported structure
# constructs are documented @ https://www.tensorflow.org/guide/data#dataset_structure.
# 
# For example, `map` can be used for adding 1 to each element, or projecting a
# subset of element components.
#
# see site-packages/tensorflow/python/data/ops/dataset_ops.py (~line 2149)
# Note this is lazy evaluated, The CB function prepare_inputs is executed only on 
# train_ds accesses

# <_MapDataset element_spec=(TensorSpec(shape=(None, 80), dtype=tf.int64, name=None), TensorSpec(shape=(None, 80), dtype=tf.int64, name=None))>
train_ds = text_ds.map(prepare_inputs)

# take(1): creates a `Dataset` with 1 element from train_ds
example_input_output = train_ds.take(1).get_single_element()

# example input
print("example input:\n", example_input_output[0][0])

# example Output (shifted by one token)
print("example (shifted) output:\n", example_input_output[1][0])





# ## 5. Create the causal attention mask function

# Causal masking is only required in decoder Transformers such as
# GPT, where the task is to sequentially generate tokens given 
# previous tokens. Masking out future tokens during training is 
# therefore essential.

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)


np.transpose(causal_attention_mask(1, 10, 10, dtype=tf.int32)[0])




# ## 6. Create a Transformer Block layer

@keras.saving.register_keras_serializable()
class TransformerBlock(keras.layers.Layer):
    # needed to add kwargs param here and pass it to base class
    # else restore model with custom classes fails...
    def __init__(self, num_heads, key_dim, embed_dim, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.attn = keras.layers.MultiHeadAttention(
            num_heads, key_dim, output_shape=embed_dim
        )
        self.dropout_1 = keras.layers.Dropout(self.dropout_rate)
        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = keras.layers.Dense(self.ff_dim, activation="relu")
        self.ffn_2 = keras.layers.Dense(self.embed_dim)
        self.dropout_2 = keras.layers.Dropout(self.dropout_rate)
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]
        causal_mask = causal_attention_mask(
            batch_size, seq_len, seq_len, tf.bool
        )
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask=causal_mask,
            return_attention_scores=True,
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return (self.ln_2(out1 + ffn_output), attention_scores)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "key_dim": self.key_dim,
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config



# ## 7. Create the Token and Position Embedding

# see https://www.tensorflow.org/guide/keras/serialization_and_saving
# see https://keras.io/guides/serialization_and_saving/
@keras.saving.register_keras_serializable()
class TokenAndPositionEmbedding(keras.layers.Layer):
    # needed to add kwargs param here and pass it to base class
    # else restore model with custom classes fails...
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        # An embedding layer is essentially a lookup table that converts each positive
        # integer token into a dense vector of length input_dim. The lookup vectors
        # are learned by the model as weights. Therefore, the number of weights
        # learned by this layer is equal to the size of the vocabulary multiplied
        # by the dimension of the embedding vector.  see keras.io/api/layers/core_layers/embedding/

        # input data must be integer encoded so each word is represented by a unique integer.
        self.token_emb = keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = keras.layers.Embedding(input_dim=max_len, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_len": self.max_len,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config

        # return {"max_len": self.max_len,
        #         "vocab_size": self.vocab_size,
        #         "embed_dim": self.embed_dim,
        #     }





# Create a TextGenerator checkpoint

# see https://www.tensorflow.org/guide/keras/writing_your_own_callbacks

class CustomCallback(keras.callbacks.Callback):
    __cbdebug = False

    def on_train_begin(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_test_begin(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Start testing; got log keys: {}".format(keys))

    def on_test_end(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Stop testing; got log keys: {}".format(keys))

    def on_predict_begin(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Start predicting; got log keys: {}".format(keys))

    def on_predict_end(self, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("Stop predicting; got log keys: {}".format(keys))

    def on_train_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Training: start of batch {}; got log keys: {}".format(batch, keys))

    def on_train_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Training: end of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Evaluating: start of batch {}; got log keys: {}".format(batch, keys))

    def on_test_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Evaluating: end of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_begin(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Predicting: start of batch {}; got log keys: {}".format(batch, keys))

    def on_predict_batch_end(self, batch, logs=None):
        keys = list(logs.keys())
        if self.__cbdebug: print("...Predicting: end of batch {}; got log keys: {}".format(batch, keys))


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, index_to_word, top_k=10):
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature):
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p=probs), probs

    def alternate_sample_from(self, logits, temperature=None):
        logits, indices = keras.metrics.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def generate(self, start_prompt, max_tokens, temperature, verbose=1):
        start_tokens = [
            self.word_to_index.get(x, 1) for x in start_prompt.split()
        ]
        sample_token = None
        info = []
        while len(start_tokens) < max_tokens and sample_token != 0:
            x = np.array([start_tokens])

            # tf_keras/src/engine/training.py - Model.predict()
            # returns y   (np.ndarray) of probabilities, size 160_000,
            #         att (np.ndarray) of attentions?, size 128
            y, att = self.model.predict(x, verbose=verbose, callbacks=[CustomCallback()])

            # pick one sample from the probabilities, this makes generated
            # text random.  temperature deterimines how random (or deteriministic)
            # the sample selection will be.  higher temperature -> more random.
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append(
                {
                    "prompt": start_prompt,
                    "word_probs": probs,
                    "atts": att[0, :, -1, :],
                }
            )
            start_tokens.append(sample_token)
            start_prompt = start_prompt + " " + self.index_to_word[sample_token]

        # wine review : italy : sicily & sardinia : red blend : [UNK] ' s nero d '
        # avola and syrah boasts remarkable popularity in its wines made to showcase
        # its earthy , bolgheri enjoys some compact , meaty quality and value characteristics
        # with raw tones of soft tannin , dusty and fresh tobacco .
        print(f"\ngenerated text:\n{start_prompt}\n")
        return info

    # see tf_keras/src/callbacks.py
    def on_epoch_end(self, epoch, logs=None):
        self.generate("wine review", max_tokens=80, temperature=1.0)



# ## 8. Build the Transformer model

if LOAD_MODEL:
    # model.load_weights('./models/model')
    gpt = keras.models.load_model("./models/gpt.keras",
                                  custom_objects = {'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
                                                    'TransformerBlock': TransformerBlock},
                                  compile=True)

    print("loaded Keras model")
    gpt.summary()

    # Tokenize starting prompt
    text_generator = TextGenerator(vocab)
    text_generator.model = gpt

else:

    # the Input layer does not need us to specify the sequence length in advance.
    # Both the batch size and the sequence length are flexible (hence the (None, None)
    # shape). This is because all downstream layers are agnostic to the length of
    # the sequence being passed through.
    #
    # shape is a tuple of integers or None objects - see keras.io/api/layers/core_layers/input/
    inputs = keras.layers.Input(shape=(None,), dtype=tf.int32)

    x = TokenAndPositionEmbedding(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)

    x, attention_scores = TransformerBlock(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM)(x)

    outputs = keras.layers.Dense(VOCAB_SIZE, activation="softmax")(x)

    gpt = keras.models.Model(inputs=inputs, outputs=[outputs, attention_scores])

    gpt.compile("adam", loss=[keras.losses.SparseCategoricalCrossentropy(), None])


    # test save and reload -- this worked
    # keras.models.save_model(gpt, "./models/custom_model.keras")

    # ??? WARNING:absl:Skipping variable loading for optimizer 'Adam', because it has 41 variables whereas
    #                  the saved optimizer has 1 variables.
    # because we never fit the model ???
    # gptreload = keras.models.load_model("./models/custom_model.keras",
    #                                     custom_objects = {'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
    #                                                       'TransformerBlock': TransformerBlock},
    #                                     compile=True)
    # print("test: loaded Keras model")


    # Model: "functional_1"
    # ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
    # ┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
    # ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
    # │ input_layer (InputLayer)        │ (None, None)           │             0 │
    # ├─────────────────────────────────┼────────────────────────┼───────────────┤
    # │ token_and_position_embedding    │ (None, None, 256)      │     2,580,480 │
    # │ (TokenAndPositionEmbedding)     │                        │               │
    # ├─────────────────────────────────┼────────────────────────┼───────────────┤
    # │ transformer_block               │ [(None, None, 256),    │       658,688 │
    # │ (TransformerBlock)              │ (None, 2, None, None)] │               │
    # ├─────────────────────────────────┼────────────────────────┼───────────────┤
    # │ dense_2 (Dense)                 │ (None, None, 10000)    │     2,570,000 │
    # └─────────────────────────────────┴────────────────────────┴───────────────┘
    #  Total params: 5,809,168 (22.16 MB)
    #  Trainable params: 5,809,168 (22.16 MB)
    #  Non-trainable params: 0 (0.00 B)

    gpt.summary()


    # ## 9. Train the Transformer

    # Create a model save checkpoint
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath="./checkpoint/checkpoint.ckpt.weights.h5",
        save_weights_only=True,
        save_freq="epoch",
        verbose=0,
    )

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")

    # Tokenize starting prompt
    text_generator = TextGenerator(vocab)

    gpt.fit(
        train_ds,
        epochs=EPOCHS,
        callbacks=[model_checkpoint_callback, tensorboard_callback, text_generator],
    )


    # Save the final model.
    # The saved .keras file contains:
    #     The model's configuration (architecture)
    #     The model's weights
    #     The model's optimizer's state (if any)
    #     Any custom classes
    keras.models.save_model(gpt, "./models/gpt.keras")




# # 3. Generate text using the Transformer

def print_probs(info, vocab, top_k=5):
    for ndx, i in enumerate(info):
        highlighted_text = []
        for word, att_score in zip(
            i["prompt"].split(), np.mean(i["atts"], axis=0)
            ):
            highlighted_text.append(
                '<span style="background-color:rgba(135,206,250,'
                + str(att_score / max(np.mean(i["atts"], axis=0)))
                + ');">'
                + word
                + "</span>"
            )
        highlighted_text = " ".join(highlighted_text)
        html = HTML(highlighted_text)
        with open("HTML/gendata_{}.html".format(ndx), "w") as file:
            file.write(html.data)

        word_probs = i["word_probs"]
        p_sorted = np.sort(word_probs)[::-1][:top_k]
        i_sorted = np.argsort(word_probs)[::-1][:top_k]
        for p, i in zip(p_sorted, i_sorted):
            print(f"{vocab[i]}:   \t{np.round(100*p,2)}%")
        print("--------\n")


# generate text from a trained model
#   provide a starting prompt to the model
#   model with predict probabilities for the next token
#   the chosen next token will be added to promot and be the next input
#
# max_tokens is the number of tokens to be generated after the prompt
info = text_generator.generate(
    "wine review : us", max_tokens=80, temperature=1.0, verbose=2
)
print(info)

info = text_generator.generate(
    "wine review : italy", max_tokens=80, temperature=0.5, verbose=2
)
print(info)

info = text_generator.generate(
    "wine review : germany", max_tokens=80, temperature=0.5, verbose=2
)
print(info)
print_probs(info, vocab)



