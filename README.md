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

## Model
The deep learning model uses convolution on top of recurrent cells. It achieves an accuracy (precision) of 85% on three genres.

## Dataset
The datset is included in the form of pickled feature vectors extracted via sonic-annotator. The original music files and categorization were collections of "best-of" music CDs.

## Dependencies
youtube-dl, sonic-annotator with plug-ins

Detailed guide in [tutorial.md](https://github.com/BSVogler/music-genre-recognition-pipeline/blob/master/Tutorial.md)

### Feature Extractions

- Vamp plugin host [sonic-annotator](https://code.soundsoftware.ac.uk/projects/sonic-annotator/files) (GNU license)
- [QM Vamp plugin](https://code.soundsoftware.ac.uk/projects/qm-vamp-plugins/files)
- [BBC Vamp plugin](https://github.com/bbcrd/bbc-vamp-plugins/releases)

On Mac OS copy the plugins to `/Library/Audio/Plug-Ins/Vamp/`.

allow execution of unsigned library in security settings after a failed attempt

### Python Dependencies
Simply run:

`sudo python3 -m pip install -r requirements.txt`
