#!/usr/bin/env python3
import argparse
import sys

args = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detects the genre of a music file.')
    parser.add_argument('filepath', help='path to file or fodler containing files')
    parser.add_argument('-k', '--keep', action='store_true', dest='keep',
                        help='if set keeps audio files')
    args = parser.parse_args()

import numpy as np
np.random.seed(1337)  # for reproducibility

import os
import re
from numpy import genfromtxt


# Parse song
if args.filepath is None:
    print("missing parameters")
    sys.exit()
filepath = args.filepath
song_folder = os.path.dirname(os.path.realpath(filepath))#should get the directory to the file

def saveToFile(genreResult):
    #if has another id save to file
    if len(sys.argv) > 2:
        id = sys.argv[2]
        if not os.path.exists("results"):
            os.makedirs("results")
        f = open('results/'+id+".txt", 'w')
        f.write(genreResult)
        f.close()

modelWeightsPath = "model_weights/merged_model_weights.hdf5";
if not os.path.exists(modelWeightsPath):
    print("No model weights found in path '"+modelWeightsPath+"'")
else:
    from split_30_seconds import batch_thirty_seconds, thirty_seconds
    from extract_features import extract_features
    
    if not os.path.exists(song_folder+"/split"):
        os.makedirs(song_folder+"/split")
        print("create folder for split file")
    
    print("Splitting file:")    
    if os.path.isdir(filepath):
        batch_thirty_seconds(song_folder)
        print("Files split. Now extracting features.")
        extract_features(song_folder)
    else:
        thirty_seconds(filepath, args.keep is None)
        print("File split. Now extracting features.")
        extract_features(song_folder+"/split/")

    from keras.models import model_from_json, Sequential
    from keras.preprocessing import sequence
    
    json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))
    import json
    model = model_from_json(json_string)
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
                  
    #mfcc coefficients
    x = []
    for root, dirs, files in os.walk(song_folder, topdown=False):
        for name in files:
            if re.search("mfcc_coefficients.csv",name):
                song_path = (os.path.join(root,name))

                song_features = genfromtxt(song_path, delimiter=",")

                if len(song_features.shape) is 2:
                    song_features = np.array([ _line[1:] for _line in song_features])
                elif len(song_features.shape) is 1:
                    song_features = np.array([song_features[1:]])

                x[0].append(song_features)

    mfcc_max_len = 0

    with( open("maxlen_mfcc_coefficients","r") ) as _f:
        mfcc_max_len = int(_f.read())

    x[0] = sequence.pad_sequences(x[0], maxlen=mfcc_max_len,dtype='float32')

    #Spectral contrast peaks
    for root, dirs, files in os.walk(song_folder, topdown=False):
        for name in files:
            if re.search("spectral-contrast_peaks.csv", name):
                song_path = (os.path.join(root,name))
                song_features = genfromtxt(song_path, delimiter=",")

                if len(song_features.shape) is 2:
                    song_features = np.array([ _line[1:] for _line in song_features])
                elif len(song_features.shape) is 1:
                    song_features = np.array([song_features[1:]])

                x[1].append(song_features)

    spectral_max_len = 0
    with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
        spectral_max_len = int(_f.read())

    x[1] = sequence.pad_sequences(x3, maxlen=spectral_max_len,dtype='float32')

    #Spectral contrast valleys
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

    x3 = sequence.pad_sequences(x[2], maxlen=spectral_max_len, dtype='float32')

    predictions = model.predict_classes(x)
    genredict = ["hiphop","pop", "rock"]
    genredict.sort()#make sure that it is alphabetically sorted
    
    #make a list of result strings
    resultsstringified = []
    for p in predictions:#p is digit
        resultsstringified.append(genredict[p])
        
    mode = max(set(resultsstringified), key=resultsstringified.count);
    
    resultstring = ""
    modeCounter=0
    for p in resultsstringified:
        if mode==p:
            modeCounter+=1
        resultstring += p+" "
        
    print("Detected "+resultstring)  
    print("The song is "+str(modeCounter*100/len(resultsstringified))+" % "+mode)
    
    saveToFile(resultstring)
