#!/usr/bin/env python3

import numpy as np
from tensorflow import keras

np.random.seed(1337)  # for reproducibility
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import LSTM, GRU, Flatten
import matplotlib.pyplot as plt
import pickle
import json

numGenres = 3


def mfcc_model(input_shape, concat):
    nb_filter = 100
    filter_length = 4
    hidden_dims = 250

    pool_length = 4

    # LSTM
    lstm_output_size = 300

    # create model
    model = keras.Input(shape=input_shape, name="mfcc")

    Convolution1D(
        input_shape=input_shape,
        filters=nb_filter,
        kernel_size=filter_length,
        padding='valid',
        strides=1)(model)
    Activation('relu')(model)
    MaxPooling1D(pool_size=pool_length)(model)
    Dropout(0.4)(model)
    Convolution1D(
        filters=int(nb_filter / 5),
        kernel_size=int(filter_length),
        padding='valid',
        strides=1)(model)
    Activation('relu')(model)
    MaxPooling1D(pool_size=pool_length)(model)
    Dropout(0.4)(model)

    LSTM(lstm_output_size,
                   # input_shape=input_shape,
                   activation='sigmoid',
                   recurrent_activation='hard_sigmoid')(model)

    Dropout(0.4)(model)
    if not concat:
        Dense(numGenres, activation='softmax')(model)

    return model


if __name__ == "__main__":
    import os

    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    # check if can write to file
    f = open('model_weights/mfcc_model_weights.hdf5', 'a')
    f.write('test')
    f.close()

    # print(X)
    # load vectorized song features
    datasetfolder = "../pickled_vectors"
    X = pickle.load(open(datasetfolder+"/mfcc_coefficients_training_vector.pickle", "rb"))
    y = pickle.load(open(datasetfolder+"/mfcc_coefficients_label.pickle", "rb"))

    X_test = pickle.load(open(datasetfolder+"/mfcc_coefficients_evaluation_training_vector.pickle", "rb"))
    y_test = pickle.load(open(datasetfolder+"/mfcc_coefficients_evaluation_label.pickle", "rb"))

    model = mfcc_model((X.shape[1], X.shape[2]), concat=False)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    print("X shape", X.shape)
    print("y shape", y.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    print("Training")

    batch_size = 60
    nb_epoch = 50
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(X_test, y_test),
                        shuffle="batch"
                        )

    with open("experimental_results.json", "w") as f:
        f.write(json.dumps(history.history, sort_keys=True, indent=4, separators=(',', ': ')))

    for k, v in history.history.items():
        # print(k,v)
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411 + _keys.index(k))
        # x_space = np.linspace(0,1,len(v))
        plt.title(k)

        plt.plot(range(0, len(v)), v, marker="8", linewidth=1.5)

    model.save_weights("model_weights/mfcc_model_weights.hdf5", overwrite=True)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('history.png')
