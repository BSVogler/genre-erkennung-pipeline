#!/bin/bash

#example url http://youtube.com/watch?v=PNjG22Gbo6U
if [ "$#" -ne 1 ]; then
	echo "pass youtube url as parameter"
else
	DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
	#cd $(dirname "$0")
	cd "$DIR"
	mkdir query
	cd query
	youtube-dl --extract-audio --audio-format mp3 -o "testfile.%(ext)s" $1 #download file
	printf "\nFile $1 downloaded \n"
	cd ..
	python3 ./querying_genre.py query/testfile.mp3 $(cut -d "=" -f 2 <<< "$1") #decode features
	rm -rf query
fi