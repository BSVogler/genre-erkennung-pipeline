# Music Genre Classification Pipeline
Everything using python:
webAPI → youtube-dl → vamp → keras → webAPI

[Demo Video](https://www.youtube.com/watch?v=fLe6uyDHeCE)

## dependencies:
keras, youtube-dl, vamp

Detailed guide in [tutorial.md](https://github.com/BSVogler/music-genre-recognition-pipeline/blob/master/Tutorial.md)

### feature extractions:

- Vamp sonic-annotator
- [QM Vamp plugin](https://code.soundsoftware.ac.uk/projects/qm-vamp-plugins/files)
- [BBC Vamp plugin](https://github.com/bbcrd/bbc-vamp-plugins/releases)

on Mac OS: 
copy plugins to /Library/Audio/Plugin-ins/Vamp

### python libraries:
- keras
- sklearn

Simply run:

`sudo python3 -m pip install keras`

`sudo python3 -m pip install sklearn`

`sudo python3 -m pip install matplotlib`
