#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Convolution1D, Activation, MaxPooling1D, Dropout, GRU
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import model_from_json, load_model

np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
import pickle
import json
import os

batch_size = 50
nb_epoch = 100
numGenres = 3

# create model
datasetpath = "../pickled_vectors"
modelarchdir = "../model_architecture/"
modelWeightsPath = "./model_weights/merged_model_weights.hdf5"


def train():
    print("creating model")
    X_1 = pickle.load(open(datasetpath + "/mfcc_coefficients_training_vector.pickle", "rb"))
    X_test_1 = pickle.load(open(datasetpath + "/mfcc_coefficients_evaluation_training_vector.pickle", "rb"))

    X_2 = pickle.load(open(datasetpath + "/spectral-contrast_peaks_training_vector.pickle", "rb"))
    X_test_2 = pickle.load(open(datasetpath + "/spectral-contrast_peaks_evaluation_training_vector.pickle", "rb"))

    print("X_1 (MFCC: items x max length x buckets)", X_1.shape)
    print("X_test_1", X_test_1.shape)
    print("X_2 (spectral contrast: items x peaks x buckets)", X_2.shape)
    print("X_test_2", X_test_2.shape)

    # crashes because input shape can not be generated using input_shapes = tf_utils.convert_shapes(inputs, to_tuples=False)
    pool_length = 4

    mfcc_shape = X_1.shape[1:]  # conv on 2d
    sc_shape = X_2.shape[1:]  # conv on 2d
    # create model
    input_mfcc = keras.Input(shape=mfcc_shape, name="mfcc_input")

    conf_mfcc = Convolution1D(
        filters=100,
        kernel_size=4,
        padding='valid',
        strides=1)(input_mfcc)
    model_mfcc = Activation('relu')(conf_mfcc)
    model_mfcc = MaxPooling1D(pool_size=pool_length)(model_mfcc)
    model_mfcc = Dropout(0.4)(model_mfcc)
    model_mfcc = Convolution1D(
        filters=20,
        kernel_size=4,
        padding='valid',
        strides=1)(model_mfcc)
    model_mfcc = Activation('relu')(model_mfcc)
    model_mfcc = MaxPooling1D(pool_size=pool_length)(model_mfcc)
    model_mfcc = Dropout(0.4)(model_mfcc)

    model_mfcc = GRU(300,
                     activation='sigmoid',
                     recurrent_activation='hard_sigmoid')(model_mfcc)

    model_mfcc = Dropout(0.4)(model_mfcc)

    inputs_scp = keras.Input(sc_shape, name="spectralconstrastpeaks")
    model = Convolution1D(
        filters=100,
        kernel_size=3,
        padding='valid',
        strides=4)(inputs_scp)
    model = Activation('relu')(model)
    model = MaxPooling1D(pool_size=1)(model)
    model = Dropout(0.2)(model)

    model_spc = GRU(100,
                    activation='sigmoid',
                    recurrent_activation='hard_sigmoid',
                    # return_sequences=True
                    )(model)

    model_spc = Dropout(0.2)(model_spc)

    merged = keras.layers.Concatenate()([model_mfcc, model_spc])
    x = keras.layers.Dense(100)(merged)
    x = keras.layers.Dense(numGenres, activation='softmax')(x)
    final_model = keras.Model(inputs=(input_mfcc, inputs_scp), outputs=x)

    final_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # write architecture to file
    savedir = modelarchdir + "merged"
    print(f"Saving architecture at {savedir}... ", end="")
    final_model.save(savedir)
    print("Ok")

    tf.keras.utils.plot_model(final_model, to_file='merged.png', show_shapes=True)

    print("Training")
    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    # check if can write to file because of the webserver might have reduced rights
    weightpath = 'model_weights/mfcc_model_weights.hdf5'
    f = open(weightpath, 'a')
    f.write('test')
    f.close()
    os.remove(weightpath)

    # #
    # final_model.load_weights("model_weights/embeddings_10_sec_split_gztan_merged_model_weights.hdf5")
    # #
    # # for i in range(10):
    # #     print("epoch",i)

    y = pickle.load(open(datasetpath + "/mfcc_coefficients_label.pickle", "rb"))
    y_test = pickle.load(open(datasetpath + "/mfcc_coefficients_evaluation_label.pickle", "rb"))
    print("y", y.shape)
    print("y_test", y_test.shape)

    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    # checkpoint
    filepath = "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = final_model.fit((X_1, X_2), y,
                              batch_size=batch_size,
                              epochs=nb_epoch,
                              validation_data=((X_test_1, X_test_2), y_test),
                              shuffle="batch",
                              callbacks=callbacks_list,
                              )

    print("Saving final result.")
    final_model.save_weights("model_weights/merged_model_weights.hdf5", overwrite=True)

    # dump training history
    with open("experimental_results.json", "w") as f:
        f.write(json.dumps(history.history, sort_keys=True, indent=4, separators=(',', ': ')))

    for k, v in history.history.items():
        _keys = list(history.history.keys())
        _keys.sort()
        plt.subplot(411 + _keys.index(k))
        plt.title(k)

        plt.plot(range(0, len(v)), v, marker="8", linewidth=1.5)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    plt.savefig('history.png')
    plot_confusion_matrix(model, y_test, x_test)


def load_evaluate():
    model = load_model(modelarchdir + "merged")
    model.load_weights(modelWeightsPath)
    y_test = pickle.load(open(datasetpath + "/mfcc_coefficients_evaluation_label.pickle", "rb"))
    x_test_1 = pickle.load(open(datasetpath + "/mfcc_coefficients_evaluation_training_vector.pickle", "rb"))
    x_test_2 = pickle.load(open(datasetpath + "/spectral-contrast_peaks_evaluation_training_vector.pickle", "rb"))
    x_test = (x_test_1, x_test_2)
    plot_confusion_matrix(model, y_test, x_test)


def plot_confusion_matrix(model, y_test, x_test):
    import sklearn.metrics
    y_pred = model.predict(x_test)
    y_pred_index = np.argmax(y_pred, axis=1)
    y_test_index = np.argmax(y_test, axis=1)
    confmx = sklearn.metrics.confusion_matrix(y_test_index, y_pred_index)
    sum_row = confmx.sum(axis=1, keepdims=True)
    confmx_norm = confmx / sum_row

    labels = ["genre0", "genre1", "genre2"]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Confusion Relative")
    cax = ax.matshow(confmx_norm, interpolation='nearest')
    fig.colorbar(cax)

    for (i, j), z in np.ndenumerate(confmx_norm):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)

    plt.show()


if __name__ == "__main__":
    train()