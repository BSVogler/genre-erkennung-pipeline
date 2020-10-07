#!/usr/bin/env python3
import argparse
import os
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
    modelarchdir = "model_architecture"

    import os
    import re
    from numpy import genfromtxt

    if keep:
        print("Keeping file because flag is set.")

    id = os.path.splitext(os.path.basename(filepath))[0]

    print("The song path: " + os.path.realpath(filepath))
    song_dir = os.path.dirname(os.path.realpath(filepath))  # should get the directory to the file
    print("The song dir is: " + song_dir)

    modelWeightsPath = "./model_weights/merged_model_weights.hdf5"
    if not os.path.exists(modelWeightsPath):
        print("No model weights found in path '" + os.path.realpath(modelWeightsPath) + "'")
    else:
        from split_30_seconds import batch_thirty_seconds, thirty_seconds
        from extract_features import extract_features

        if not os.path.exists(song_dir + "/split" + id):
            os.makedirs(song_dir + "/split" + id)
            print("create folder " + song_dir + "/split" + id + " for split file parts")

        if os.path.isdir(filepath):
            print("Splitting files in folder")
            batch_thirty_seconds(song_dir)
            print("Now extracting features.")
            extract_features(song_dir)
        else:
            print("Splitting file: " + filepath)
            thirty_seconds(song_dir + "/" + os.path.basename(filepath), not keep)
            if not os.path.isfile(
                    song_dir + "/split" + id + "/000_vamp_bbc-vamp-plugins_bbc-spectral-contrast_peaks.csv"):
                print("Now extracting features.")
                extract_features(song_dir + "/split" + id + "/")
            else:
                print("Skipping feature extraction because feature file was found.")
        from keras.models import model_from_json
        from keras.preprocessing import sequence

        import json
        with open(modelarchdir+"/merged_model_architecture.json", "r") as modelfile:
            json_string = json.load(modelfile)
        model = model_from_json(json_string)
        model.load_weights(modelWeightsPath)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy']
                      )


        # mfcc coefficients
        vector_mfcc = []
        for root, dirs, files in os.walk(song_dir, topdown=False):
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
        with(open(modelarchdir+"maxlen.json", "r")) as _f:
            maxvalues = json.load(_f)


        x = []
        x.append(sequence.pad_sequences(vector_mfcc, maxlen=maxvalues["mfcc_coefficients"], dtype='float32'))

        # Spectral contrast peaks
        vectorSCP = []
        for root, dirs, files in os.walk(song_dir, topdown=False):
            for name in files:
                if re.search("spectral-contrast_peaks.csv", name):
                    song_path = (os.path.join(root, name))
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) == 2:
                        song_features = np.array([_line[1:] for _line in song_features])
                    elif len(song_features.shape) == 1:
                        song_features = np.array([song_features[1:]])

                    vectorSCP.append(song_features)

        x.append(sequence.pad_sequences(vectorSCP, maxlen=maxvalues["spectral-contrast_peaks"], dtype='float32'))

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


        x[2] = sequence.pad_sequences(x[2], maxlen=maxvalues["spectral-contrast_peaks"], dtype='float32')'''

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

    songpath = os.path.realpath(args.filepath) #get absolute path before changing cwd
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    query(songpath, args.keep is not None)
