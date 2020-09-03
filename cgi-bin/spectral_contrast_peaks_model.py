#!/usr/bin/env python3
import numpy as np
from tensorflow import keras

np.random.seed(1337)  # for reproducibility
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation
from tensorflow.python.keras.layers import Convolution1D, MaxPooling1D
from tensorflow.python.keras.layers import LSTM
import matplotlib.pyplot as plt
import pickle

numGenres = 3


# load vectorized song features
#
def model(input_shape, concat=False):
    """

    :param input_shape:
    :param concat: when should return whole model use False, if it will be concatenated use True
    :return:
    """
    nb_filter = 100
    filter_length = 3
    hidden_dims = 250
    pool_length = 1

    # LSTM
    lstm_output_size = 100

    # print("creating model")
    # create model
    model = keras.Input(input_shape, name="spectralconstrastpeaks")
    Convolution1D(
        input_shape=input_shape,
        filters=nb_filter,
        kernel_size=filter_length,
        padding='valid',
        strides=4)(model)
    Activation('relu')(model)
    MaxPooling1D(pool_size=pool_length)(model)
    Dropout(0.2)(model)

    LSTM(lstm_output_size,
         # input_shape=input_shape,
         activation='sigmoid',
         recurrent_activation='hard_sigmoid',
         # return_sequences=True
         )(model)

    Dropout(0.2)(model)

    if not concat:
        Dense(numGenres, activation='softmax')(model)

    return model


if __name__ == "__main__":
    datasetfolder = "../pickled_vectors"
    X = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_training_vector.pickle", "rb"))
    y = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_label.pickle", "rb"))

    X_test = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_evaluation_training_vector.pickle", "rb"))
    y_test = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_evaluation_label.pickle", "rb"))

    model = model((X.shape[1], X.shape[2]), concat=False)
    # model = keras.layers.Dense(numGenres, activation='softmax')(model)
    # model = keras.model.Model(inputs=[input1], outputs=out)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )

    # print(X)
    print("X shape", X.shape)
    print("y shape", y.shape)
    print("X_test", X_test.shape)
    print("y_test", y_test.shape)

    print("Training")
    #
    # X = np.random.random(X.shape)
    # y = np.random.random(y.shape)
    #
    batch_size = 70
    nb_epoch = 50
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=nb_epoch,
                        validation_data=(X_test, y_test),
                        shuffle=True
                        )
    #
    # # with open("experimental_results.json","w") as f:
    # #     f.write(json.dumps(history.history, sort_keys=True,indent=4, separators=(',', ': ')))
    #
    import os

    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")
    model.save_weights("model_weights/spectral_contrast_peaks_model_weights.hdf5", overwrite=True)
    for k, v in history.history.items():
        # print(k,v)
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411 + _keys.index(k))
        # x_space = np.linspace(0,1,len(v))
        plt.title(k)

        plt.plot(range(0, len(v)), v, marker="8", linewidth=1.5)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('history.png')
