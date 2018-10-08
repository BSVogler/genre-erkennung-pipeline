#!/usr/bin/env python3
import numpy as np

np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense, Concatenate
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import pickle
import json
import mfcc_model
import spectral_contrast_peaks_model
import os

if __name__ == "__main__":
    batch_size = 50
    nb_epoch = 300
    numGenres = 3

    print("creating model")
    # create model
    datasetfolder = "../pickled_vectors"

    X_1 = pickle.load(open(datasetfolder + "/mfcc_coefficients_training_vector.pickle", "rb"))
    X_test_1 = pickle.load(open(datasetfolder + "/mfcc_coefficients_evaluation_training_vector.pickle", "rb"))

    X_2 = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_training_vector.pickle", "rb"))
    X_test_2 = pickle.load(open(datasetfolder + "/spectral-contrast_peaks_evaluation_training_vector.pickle", "rb"))

    model_1 = mfcc_model.mfcc_model((X_1.shape[1], X_1.shape[2]), True)
    model_2 = spectral_contrast_peaks_model.model((X_2.shape[1], X_2.shape[2]), True)

    # print("X_1",X_1.shape)
    # print("X_test_1",X_test_1.shape)
    print("X_1", X_1.shape)
    print("X_test_1", X_test_1.shape)
    print("X_2", X_2.shape)
    print("X_test_2", X_test_2.shape)

    merged = Concatenate([model_1, model_2])
    third = Sequential()
    third.add(merged)
    third.add(Dense(100))
    third.add(Dense(numGenres, activation='softmax'))
    final_model = third
    #final_model = Concatenate([merged,  third])

    final_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    from keras.utils import plot_model

    plot_model(model_1, to_file='model1.png')
    plot_model(model_2, to_file='model2.png')
    plot_model(final_model, to_file='merged.png')
    # plot(model_1,to_file="model_1.png")
    # plot(model_1,to_file="model_1.png")
    # plot(final_model,to_file="merged_model.png")

    # write architecture to file
    #crashes here, also it is not trainable
    json_string = final_model.to_json()
    with open("../model_architecture/merged_model_architecture.json", "w") as f:
        f.write(json.dumps(json_string, sort_keys=True, indent=4, separators=(',', ': ')))

    print("Training")
    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    # check if can write to file
    f = open('model_weights/mfcc_model_weights.hdf5', 'a')
    f.write('test')
    f.close()
    # #
    # final_model.load_weights("model_weights/embeddings_10_sec_split_gztan_merged_model_weights.hdf5")
    # #
    # # for i in range(10):
    # #     print("epoch",i)

    y = pickle.load(open(datasetfolder + "/mfcc_coefficients_label.pickle", "rb"))
    y_test = pickle.load(open(datasetfolder + "/mfcc_coefficients_evaluation_label.pickle", "rb"))
    print("y", y.shape)
    print("y_test", y_test.shape)

    if not os.path.exists("model_weights"):
        os.makedirs("model_weights")

    # checkpoint
    filepath = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    history = final_model.fit([X_1, X_2], y,
                              batch_size=batch_size,
                              nb_epoch=nb_epoch,
                              validation_data=([X_test_1, X_test_2], y_test),
                              shuffle="batch",
                              callbacks=callbacks_list,
                              )
    print("saving final result")
    final_model.save_weights("model_weights/merged_model_weights.hdf5", overwrite=True)

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
