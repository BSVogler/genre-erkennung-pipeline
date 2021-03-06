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

#example: http://youtube.com/watch?v=PNjG22Gbo6U
arguments = cgi.FieldStorage()
if len(arguments) >= 1:
    
    youtubeURL = str(arguments["youtubeurl"].value)
    rootPath = "./"
    queryPath = rootPath+"query"
    if not os.path.exists(queryPath):
        os.makedirs(queryPath)
    
    youtubeID = youtubeURL[youtubeURL.find("=")+1:]
    #print(youtubeID)
    
    #mark as waiting
    if os.path.exists(rootPath+"results/"+youtubeID+".txt"):
        with open(rootPath+"results/"+youtubeID+".txt",'r') as resultfile:
            print(resultfile.read())
    else:
        with open(rootPath+"results/"+youtubeID+".txt",'w') as resultfile:
            resultfile.write("pending")
            
        #download file
        if not os.path.exists(queryPath+"/"+youtubeID+".mp3"):    
            print("Downloading "+youtubeURL+"<br>")
            cmd = "youtube-dl --extract-audio --audio-format mp3 -o "+queryPath+"/"+youtubeID+".%(ext)s "+youtubeID
            subprocess.call(cmd.split())
            print("Downloaded")
        else:
            print("mp3 file already downloaded")
        #extract features and query
        import querying_genre as qg
        qg.query(queryPath+"/"+youtubeID+".mp3", True)

        #remove -rf "query"
        # p = Popen(cmd.split(), stdin=PIPE, stdout=PIPE, stderr=PIPE)
        #output, err = p.communicate()
        print("Request accepted: "+youtubeID+"<br>")    
            
else:
    print("missing or empty argument <i>youtubeurl</i>")
