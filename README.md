# Music Genre Classification Pipeline
With some keras API changes some stuff broke. The current combined model does not work. The loading of the architecture does not work, so it has to be rebuild and trained again.


Everything using python:
webAPI → youtube-dl → vamp → keras → webAPI

[Demo Video](https://www.youtube.com/watch?v=fLe6uyDHeCE)

## dependencies:
youtube-dl, vamp

Detailed guide in [tutorial.md](https://github.com/BSVogler/music-genre-recognition-pipeline/blob/master/Tutorial.md)

### feature extractions:

- Vamp sonic-annotator
- [QM Vamp plugin](https://code.soundsoftware.ac.uk/projects/qm-vamp-plugins/files)
- [BBC Vamp plugin](https://github.com/bbcrd/bbc-vamp-plugins/releases)

on Mac OS: 
copy plugins to /Library/Audio/Plugin-ins/Vamp

### python libraries:
Simply run:

`sudo python3 -m pip install -r requirements.txt`
