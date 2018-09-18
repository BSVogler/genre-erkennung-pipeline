#!/usr/bin/env python3
import subprocess
from subprocess import Popen, PIPE
import sys
from split_30_seconds import iterate_audio
import os

audio_features = [
                    #   "vamp:qm-vamp-plugins:qm-tempotracker:tempo",
                      "vamp:qm-vamp-plugins:qm-mfcc:coefficients",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:peaks",
                      "vamp:bbc-vamp-plugins:bbc-spectral-contrast:valleys",
                      ]
                      
def extract_features(path="."):
    for feature in audio_features:
        cmd = "sonic-annotator", "-d", feature, path, "-r", "-w", "csv", "--csv-force"
        subprocess.call(cmd)
        #p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        #output, err = p.communicate()

def extract_features_single(path="."):
    for feature in audio_features:
        cmd = "sonic-annotator", "-d", feature, path, "-w", "csv", "--csv-force"
        subprocess.call(cmd)

if __name__=="__main__":
    # extract_features("dataset/gztan_split_10sec")

    #    extract_features_single(audio_path)
    # extract_features("dataset/train/hiphop")
    # extract_features("dataset/train/rock")
    # extract_features("dataset/train/pop")

    # extract_features("dataset/train")
    if len(sys.argv) < 2:
        print("missing parameter for dataset or file path")
    else:
        if os.path.isdir(sys.argv[1]):
            for file in iterate_audio(path=sys.argv[1]):
                print("extracing" + file)
                extract_features_single(file)
        else:
            extract_features_single(sys.argv[1])
    
    #extract features for single file
