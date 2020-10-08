#!/usr/local/bin/python3
import json
import os
import re
from typing import Tuple

import numpy as np
import pickle
import sys
from keras.preprocessing import sequence
from keras.utils import np_utils

"""
A class used to parse the feature csv files for later processing with numpy.
"""

def vectorize_song_feature(filepath):
    song_features = np.genfromtxt(filepath, delimiter=",")


def create_dataset(dataset_path: str, feature=None, lower_limit=None, upper_limit=None, verbose=False,
                   categorical=True) -> Tuple:
    """
    Obtain numpy vector from csv datapoints.
    :param dataset_path:
    :param feature:
    :param lower_limit:
    :param upper_limit:
    :param verbose:
    :param categorical:
    :return:
    """
    training_vector = []
    labels = []

    if not os.path.isdir(dataset_path):
        print("dataset could not be found")

    # start with lower dirs
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        genres = [genre for genre in dirs]
    print("genres", genres)
    idx = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        if root != dataset_path:  # ignore top level
            genre = os.path.basename(root)
            print("Processing: " + genre)
            for name in files:
                # identify the feature files
                if re.search(feature + ".csv", name):
                    song_path = os.path.join(root, name)
                    if verbose:
                        print(song_path)

                    song_features = np.genfromtxt(song_path, delimiter=",")
                    song_features = song_features[..., lower_limit:upper_limit]
                    training_vector.append(song_features)
                    labels.append(idx)
            idx += 1

    if categorical:
        labels = np_utils.to_categorical(labels, num_classes=len(genres))
    maxlen = np.max([len(vector) for vector in training_vector])
    return training_vector, labels, maxlen


def build_vectors(feature="", data_label="", lower_limit=None, upper_limit=None, dir_path: str = "dataset"):
    """
    write training vectors to pickle files
    :param feature:
    :param data_label:
    :param lower_limit:
    :param upper_limit:
    :param dir_path:
    :return:
    """
    # training
    training_vector, labels, maxlen_training = create_dataset(dataset_path=dir_path + "/train", feature=feature,
                                                              lower_limit=lower_limit, upper_limit=upper_limit)

    # validation
    evaluation_training_vector, evaluation_labels, maxlen_evaluation = create_dataset(
        dataset_path=dir_path + "/test",
        feature=feature,
        lower_limit=lower_limit,
        upper_limit=upper_limit
    )

    # X_training
    maxlen = np.max([maxlen_training, maxlen_evaluation])
    training_vector = sequence.pad_sequences(training_vector, maxlen=maxlen, dtype='float32')
    # write to file
    training_file = open(pickledir + f"{data_label}{feature}_training_vector.pickle", "wb")
    pickle.dump(training_vector, training_file)
    # write y
    label_file = open(pickledir + f"{data_label}{feature}_label.pickle", "wb")
    pickle.dump(labels, label_file)

    # evaluation
    evaluation_training_vector = sequence.pad_sequences(evaluation_training_vector,
                                                        maxlen=maxlen,
                                                        dtype='float32')
    evalFile = open(pickledir + f"{data_label}{feature}_evaluation_training_vector.pickle", "wb")
    pickle.dump(evaluation_training_vector, evalFile)

    # # evaluation
    pickle.dump(evaluation_labels,
                open(pickledir + f"{data_label}{feature}_evaluation_label.pickle", "wb"))

    maxlendict[feature] = int(maxlen)
    with open(pickledir + "maxlen.json", 'w') as f:
        json.dump(maxlendict, f)


if __name__ == "__main__":
    """Pass parameter to dataset via cli."""
    if len(sys.argv) < 2:
        print("missing parameter for dataset path")
    else:
        path = sys.argv[1]
        pickledir = "../pickled_vectors/"
        if not os.path.exists(pickledir):
            os.makedirs(pickledir)
        maxlendict = {}
        build_vectors(dir_path=path, feature="spectral-contrast_peaks", lower_limit=1)
        build_vectors(dir_path=path, feature="mfcc_coefficients", lower_limit=1)
        # build_vectors(keyword="tempotracker_tempo",upper_limit=-1)
        # create_dataset("dataset/my_dataset",keyword="spectral-contrast_peaks",lower_limit=1)
        # create_dataset("dataset/my_dataset",keyword="mfcc_coefficients",lower_limit=1)
