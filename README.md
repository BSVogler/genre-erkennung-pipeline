# Music Genre Classification Pipeline
The pipeline is managed with python. The webAPI can be accessed with a web view.
```
webAPI → youtube-dl
                 ↓
webAPI ← keras ← vamp
```

[Demo Video](https://www.youtube.com/watch?v=fLe6uyDHeCE)


Current known issues:
The webserver does not serve python files.


## dependencies:
youtube-dl, vamp

Detailed guide in [tutorial.md](https://github.com/BSVogler/music-genre-recognition-pipeline/blob/master/Tutorial.md)

### feature extractions:

- Vamp plugin host [sonic-annotator](https://code.soundsoftware.ac.uk/projects/sonic-annotator/files) (GNU license)
- [QM Vamp plugin](https://code.soundsoftware.ac.uk/projects/qm-vamp-plugins/files)
- [BBC Vamp plugin](https://github.com/bbcrd/bbc-vamp-plugins/releases)

on Mac OS: 
copy plugins to `/Library/Audio/Plug-Ins/Vamp/`

allow execution of unsigned library in security settings after a failed attempt

### python libraries:
Simply run:

`sudo python3 -m pip install -r requirements.txt`
