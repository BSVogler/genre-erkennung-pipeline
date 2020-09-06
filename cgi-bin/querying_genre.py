#!/usr/bin/env python3
import argparse
import sys

args = []


def query(filepath, keep=True):
    """

    :param filepath:
    :param keep:
    :return:
    """
    import numpy as np
    np.random.seed(1337)  # for reproducibility

    import os
    import re
    from numpy import genfromtxt

    if keep:
        print("Keeping file because flag is set.")

    id = os.path.splitext(os.path.basename(filepath))[0]

    print("The song path: " + filepath)
    song_folder = os.path.dirname(os.path.realpath(filepath))  # should get the directory to the file
    print("The song folder is: " + song_folder)

    modelWeightsPath = "./model_weights/merged_model_weights.hdf5"
    if not os.path.exists(modelWeightsPath):
        print("No model weights found in path '" + os.path.dirname(os.path.realpath(modelWeightsPath)) + "'")
    else:
        from split_30_seconds import batch_thirty_seconds, thirty_seconds
        from extract_features import extract_features

        if not os.path.exists(song_folder + "/split" + id):
            os.makedirs(song_folder + "/split" + id)
            print("create folder " + song_folder + "/split" + id + " for split file parts")

        if os.path.isdir(filepath):
            print("Splitting files in folder")
            batch_thirty_seconds(song_folder)
            print("Now extracting features.")
            extract_features(song_folder)
        else:
            print("Splitting file: " + filepath)
            thirty_seconds(song_folder + "/" + os.path.basename(filepath), not keep)
            if not os.path.isfile(
                    song_folder + "/split" + id + "/000_vamp_bbc-vamp-plugins_bbc-spectral-contrast_peaks.csv"):
                print("Now extracting features.")
                extract_features(song_folder + "/split" + id + "/")
            else:
                print("Skipping feature extraction because feature file was found.")
        from keras.models import model_from_json, Sequential
        from keras.preprocessing import sequence

        import json
        with open("model_architecture/merged_model_architecture.json", "r") as modelfile:
            json_string = json.load(modelfile)
        model = model_from_json(json_string)
        model.load_weights(modelWeightsPath)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy']
                      )


        # mfcc coefficients
        vector_mfcc = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search("mfcc_coefficients.csv", name):
                    song_path = (os.path.join(root, name))

                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) == 2:
                        song_features = np.array([_line[1:] for _line in song_features])
                    elif len(song_features.shape) == 1:
                        song_features = np.array([song_features[1:]])

                    vector_mfcc.append(song_features)

        mfcc_max_len = 0
        with(open("maxlen_mfcc_coefficients", "r")) as _f:
            mfcc_max_len = int(_f.read())

        x = []
        x.append(sequence.pad_sequences(vector_mfcc, maxlen=mfcc_max_len, dtype='float32'))

        # Spectral contrast peaks
        vectorSCP = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search("spectral-contrast_peaks.csv", name):
                    song_path = (os.path.join(root, name))
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) == 2:
                        song_features = np.array([_line[1:] for _line in song_features])
                    elif len(song_features.shape) == 1:
                        song_features = np.array([song_features[1:]])

                    vectorSCP.append(song_features)

        spectral_max_len = 0
        with(open("maxlen_spectral-contrast_peaks", "r")) as _f:
            spectral_max_len = int(_f.read())

        x.append(sequence.pad_sequences(vectorSCP, maxlen=spectral_max_len, dtype='float32'))

        # Spectral contrast valleys
        '''x.append([])
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search("spectral-contrast_valleys.csv",name):
                    song_path = (os.path.join(root,name))
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x[2].append(song_features)

        spectral_max_len = 0
        with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
            spectral_max_len = int(_f.read())

        x[2] = sequence.pad_sequences(x[2], maxlen=spectral_max_len, dtype='float32')'''

        predictions = model.predict_classes(x)
        genredict = ["hiphop", "pop", "rock"]

        # transform one hot encoding to human readable format
        resultsstringified = []
        resultStringList = ""
        for p in predictions:  # p is digit
            resultsstringified.append(genredict[p])
            resultStringList += genredict[p] + " "

        genreMaxResult = max(set(resultsstringified), key=resultsstringified.count)

        modeCounter = 0
        for p in resultsstringified:
            if genreMaxResult == p:
                modeCounter += 1  # count percentage of maximum result

        print("Detected " + resultStringList)
        endresultString = "The song is " + "{:3.2f}".format(
            modeCounter * 100 / len(resultsstringified)) + " % " + genreMaxResult
        print(endresultString)

        def saveToFile(genreResult):
            # if has another id save to file
            if not os.path.exists("results"):
                os.makedirs("results")
            f = open('results/' + id + ".txt", 'w')
            f.write(genreResult)
            f.close()

        saveToFile(genreResult=endresultString)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detects the genre of a music file.')
    parser.add_argument('filepath', help='path to file or folder containing files')
    parser.add_argument('-k', '--keep', action='store_true', dest='keep',
                        help='if set keeps audio files')
    args = parser.parse_args()
    # Parse song
    if args.filepath is None:
        print("missing parameters")
        sys.exit()
    query(args.filepath, args.keep is not None)
