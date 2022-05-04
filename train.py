import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import re, random
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import losses, callbacks, utils, models, Input
from tensorflow.keras import layers as L

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

from config import config
from data_preprocessing import data_preprocessing, create_dataset, preprocess_input
from transformers import *



run = neptune.init(
    project="nielspace/machine-translation",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjRhYzI0Ny0zZjBmLTQ3YjYtOTY0Yi05ZTQ4ODM3YzE0YWEifQ==",
)  

params = {
    "VALIDATION_SIZE":config.VALIDATION_SIZE,
    "BATCH_SIZE":config.BATCH_SIZE,
    "MAX_EPOCHS":config.MAX_EPOCHS,
    "EMBED_DIM":config.EMBED_DIM,
    "DENSE_DIM":config.DENSE_DIM,
    "NUM_HEADS":config.NUM_HEADS,
    "X_LEN":config.X_LEN,
    "Y_LEN":config.Y_LEN,
    "NUM_ENCODER_TOKENS":config.NUM_ENCODER_TOKENS,
    "NUM_DECODER_TOKENS":config.NUM_DECODER_TOKENS,
    "OPTIMIZER":config.OPTIMIZER,
    "LOSS":config.LOSS,
    "METRICS":config.METRICS
}
run["parameters"] = params


input_texts, target_texts = data_preprocessing('./data/fra.txt')

data = create_dataset(input_texts, target_texts, len(input_texts))

english = Tokenizer(char_level=True)
french = Tokenizer(char_level=True)

english.fit_on_texts(data.English.values)
french.fit_on_texts(data.French.values)

X = np.array(list(map(lambda x: preprocess_input(x, english, config.X_LEN), data['English'])))
y = np.array(list(map(lambda x: preprocess_input(x, french, config.Y_LEN), data['French'])))


X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=config.VALIDATION_SIZE, random_state=23)


def generate_batch(X, y, batch_size=config.BATCH_SIZE):
    ''' Generate a batch of data '''
    while True:
        for j in range(0, len(X), batch_size):
            encoder_input_data = X[j:j+batch_size]
            decoder_input_data = y[j:j+batch_size]
            output = y[j:j+batch_size]
            decoder_output_data = np.zeros_like(output)
            decoder_output_data[:,:-1] = output[:, 1:]
            decoder_target_data = utils.to_categorical(decoder_output_data, num_classes=64)
            yield([encoder_input_data, decoder_input_data], decoder_target_data)

encoder_inputs = keras.Input(shape=(None, ), dtype="int64", name="English")
x = PositionalEmbedding(config.X_LEN, config.NUM_ENCODER_TOKENS, config.EMBED_DIM)(encoder_inputs)
encoder_outputs = TransformerEncoder(config.EMBED_DIM, config.DENSE_DIM, config.NUM_HEADS)(x)

decoder_inputs = keras.Input(shape=(None, ), dtype="int64", name="French" )
x = PositionalEmbedding(config.Y_LEN, config.NUM_DECODER_TOKENS, config.EMBED_DIM)(decoder_inputs)
x = TransformerDecoder(config.EMBED_DIM, config.DENSE_DIM, config.NUM_HEADS)(x, encoder_outputs)
x = L.Dropout(0.5)(x)
decoder_outputs = L.Dense(config.NUM_DECODER_TOKENS, activation="softmax")(x)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile(optimizer=config.OPTIMIZER, loss=config.LOSS, metrics=config.METRICS)

es = callbacks.EarlyStopping(monitor="val_loss", patience=3, verbose=3, restore_best_weights=True, min_delta=1e-4)
rlp = callbacks.ReduceLROnPlateau(monitor='val_loss', patience=2, verbose=3)

neptunecallback = NeptuneCallback(run=run, base_namespace="metrics")

history = transformer.fit(
    generate_batch(X_train, y_train), steps_per_epoch = np.ceil(len(X_train)/config.BATCH_SIZE),
    validation_data=generate_batch(X_valid, y_valid), validation_steps=np.ceil(len(X_valid)/config.BATCH_SIZE),
    epochs=2, callbacks=[neptunecallback, es, rlp])

history
run.stop()
print('done')