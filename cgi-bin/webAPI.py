#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
#written by Benedikt Vogler

import cgi
import cgitb
import random
from subprocess import Popen, PIPE
import subprocess
import os.path

cgitb.enable()

print("Content-Type: text/html")     # HTML is following
print()  
                             # blank line, end of headers

arguments = cgi.FieldStorage()
if len(arguments) >= 1:
    
    youtubeURL = str(arguments["youtubeurl"].value)
    rootPath = "./"
    if not os.path.exists(rootPath+"query"):
        os.makedirs(rootPath+"query")
    #download file
    print("Downloading "+youtubeURL+"<br>")
    #cmd = "youtube-dl --extract-audio --audio-format mp3 -o "+rootPath+"query/testfile.%(ext)s "+youtubeID
    #subprocess.call(cmd.split())
    print("Downloaded")
        #print("\nFile $1 downloaded \n")
        #extract features
        
    youtubeID=youtubeURL[youtubeURL.find("=")+1:]
    print(youtubeID)
    
    import querying_genre as qg
    qg.query(youtubeID)
    #cmd = "python3 ./querying_genre.py query/testfile.mp3 $(cut -d "=" -f 2 <<< "+youtubeID+") #decode features"
    #subprocess.call(cmd.split())
    #remove -rf "query"
   # p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
    #output, err = p.communicate()
    print("Request accepted: "+youtubeID)
else:
    print("missing or empty argument <i>youtubeurl</i>")
