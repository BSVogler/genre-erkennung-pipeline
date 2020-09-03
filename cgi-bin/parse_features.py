#!/usr/local/bin/python3
import os
import re
import numpy as np
import pickle
import sys
from numpy import genfromtxt
from keras.preprocessing import sequence
from keras.utils import np_utils


# this is probably not the best way to encode
# a one-hot label from the song path

def encode_label(genres, song_path, verbose=True):
    label = None
    for idx, genre in enumerate(genres):
        if re.search(genre, song_path):
            label = idx
            # print(genre)
            if verbose:
                print(song_path)
            # print(label)
            break
    return label


def vectorize_song_feature(filepath):
    song_features = genfromtxt(filepath, delimiter=",")


def create_dataset(dataset_path, keyword=None, lower_limit=None, upper_limit=None, verbose=True, categorical=True):
    training_vector = []
    labels = []

    for root, dirs, files in os.walk(dataset_path, topdown=False):
        genres = [_dir for _dir in dirs]
        # print(genres)
    print("genres", genres)
    # print(dataset_path)
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for name in files:
            if re.search(keyword+".csv", name):
                song_path = (os.path.join(root, name))
                if verbose:
                    print(song_path)

                song_features = genfromtxt(song_path, delimiter=",")

                if len(song_features.shape) is 2:
                    song_features = np.array([_line[lower_limit:upper_limit] for _line in song_features])
                elif len(song_features.shape) is 1:
                    song_features = np.array([song_features[lower_limit:upper_limit]])
                training_vector.append(song_features)

                labels.append(encode_label(genres, song_path, verbose=verbose))
                # print(encode_label(genres,song_path))

    if categorical:
        # print(labels)
        labels = np_utils.to_categorical(labels)
    maxlen = np.max([len(vector) for vector in training_vector])
    return training_vector, labels, maxlen


def build_vectors(keyword="", data_label="", lower_limit=None, upper_limit=None, folder_path="dataset"):
    # training
    training_vector, labels, maxlen_training = create_dataset(dataset_path=folder_path + "/train", keyword=keyword,
                                                              lower_limit=lower_limit, upper_limit=upper_limit)

    # validation
    evaluation_training_vector, evaluation_labels, maxlen_evaluation = create_dataset(
        dataset_path=folder_path+"/test", keyword=keyword, lower_limit=lower_limit, upper_limit=upper_limit)

    # # X_training
    maxlen = np.max([maxlen_training, maxlen_evaluation])
    training_vector = sequence.pad_sequences(training_vector, maxlen=maxlen, dtype='float32')
    # write to file
    training_file = open(f"pickled_vectors/{data_label}{keyword}_training_vector.pickle","wb")
    pickle.dump(training_vector, training_file)
    # write y
    label_file = open(f"pickled_vectors/{data_label}{keyword}_label.pickle", "wb")
    pickle.dump(labels, label_file)

    # evaluation
    evaluation_training_vector = sequence.pad_sequences(evaluation_training_vector,
                                                        maxlen=maxlen,
                                                        dtype='float32')
    evalFile = open(f"pickled_vectors/{data_label}{keyword}_evaluation_training_vector.pickle", "wb")
    pickle.dump(evaluation_training_vector, evalFile)

    # # evaluation
    pickle.dump(evaluation_labels,
                open(f"pickled_vectors/{data_label}{keyword}_evaluation_label.pickle", "wb"))
    with(open(f"maxlen_{keyword}", "w")) as f:
        f.write(str(maxlen))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("missing parameter for dataset path")
    else:
        path = sys.argv[1]
        if not os.path.exists("pickled_vectors"):
            os.makedirs("pickled_vectors")
        build_vectors(folder_path=path, keyword="spectral-contrast_peaks", lower_limit=1)
        build_vectors(folder_path=path, keyword="mfcc_coefficients", lower_limit=1)
        # build_vectors(keyword="tempotracker_tempo",upper_limit=-1)
        # create_dataset("dataset/my_dataset",keyword="spectral-contrast_peaks",lower_limit=1)
        # create_dataset("dataset/my_dataset",keyword="mfcc_coefficients",lower_limit=1)
