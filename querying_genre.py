#!/usr/local/bin/python3
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import model_from_json, Sequential
import json
from extract_features import extract_features
from split_30_seconds import batch_thirty_seconds, thirty_seconds
import os
import re
import sys
from numpy import genfromtxt
from keras.preprocessing import sequence

def saveToFile(genreResult):
    #if has another id save to file
    if len(sys.argv) > 2:
        id = sys.argv[2]
        if not os.path.exists("results"):
            os.makedirs("results")
        f = open('results/'+id+".txt", 'w')
        f.write(genreResult)
        f.close()

path = "model_weights/merged_model_weights.hdf5";
if not os.path.exists(path):
    print("No model weights found in path '"+path+"'")
else:
    json_string = json.load(open("model_architecture/merged_model_architecture.json","r"))
    model = model_from_json(json_string)
    model.load_weights(path)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy']
                  )
    # Parse song
    if len(sys.argv) < 2:
        print("missing parameter")
    else:
        filename = sys.argv[1]
        song_folder = os.path.dirname(os.path.realpath(filename))#should get the directory to the file

        if os.path.isdir(filename):
            batch_thirty_seconds(song_folder)
            extract_features(song_folder)
        else:
            thirty_seconds(filename)
            print("File split. Now extracting features.")
            extract_features(song_folder)
            print("Extracted features.")
             
        keyword_2 = "mfcc_coefficients"

        x2 = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search(keyword_2+".csv",name):
                    song_path = (os.path.join(root,name))

                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x2.append(song_features)

        mfcc_max_len = 0

        with( open("maxlen_mfcc_coefficients","r") ) as _f:
            mfcc_max_len = int(_f.read())

        x2 = sequence.pad_sequences(x2, maxlen=mfcc_max_len,dtype='float32')

        keyword_3 = "spectral-contrast_peaks"
        x3 = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search(keyword_3+".csv",name):
                    song_path = (os.path.join(root,name))
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x3.append(song_features)

        spectral_max_len = 0
        with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
            spectral_max_len = int(_f.read())

        x3 = sequence.pad_sequences(x3, maxlen=spectral_max_len,dtype='float32')

        keyword_4 = "spectral-contrast_valleys"
        x4 = []
        for root, dirs, files in os.walk(song_folder, topdown=False):
            for name in files:
                if re.search(keyword_4+".csv",name):
                    song_path = (os.path.join(root,name))
                    song_features = genfromtxt(song_path, delimiter=",")

                    if len(song_features.shape) is 2:
                        song_features = np.array([ _line[1:] for _line in song_features])
                    elif len(song_features.shape) is 1:
                        song_features = np.array([song_features[1:]])

                    x4.append(song_features)

        spectral_max_len = 0
        with( open("maxlen_spectral-contrast_peaks","r") ) as _f:
            spectral_max_len = int(_f.read())

        x4 = sequence.pad_sequences(x4, maxlen=spectral_max_len,dtype='float32')

        predictions = model.predict_classes([x2,x3,x4])
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

