*This is a work in progress readme. Please refer to README_original.md if you would like to learn more about the full capabilities of this repo.
The following readme is only what is required to get testing/test_capture.py working.*

*This is a work in progress readme that only covers some of the functionality offered by this repo. Please refer to README_original.md if you would to learn more about the full capabilities of this repo.*

### Required packages
The required packages are listed in `requirements.txt`. I recommend installing these in a `virtualenv` so as to leave your normal packages untouched. Some of the package version (e.g. for tensorflow) are quite outdated so you would not want that to mess with your current setup.


### Added functionality
Beyond what the original repo could initially do, this fork has added the following functionality:
* Run a detector continously on fixed location of your screen and then display any detected boxes and landmarks within a separate window.
* Slowly fade in pixels of an image to investigate when the critical number of pixels have appeared such that a detection is triggered.

### Continous Detector
#### Paths that require changing
Please change the following paths:
- In `testing/mark_detector`, navigate to the `__init__` of the `FaceDetector`. You will see two file paths listed. Change these to match the filepaths on your machine.
- In `testing/test_capture`, at the top of the code and in the `__main__` function, you will see two file paths listed. Change these to match the filepaths on your machine.
For both of these, everything after the repo name already exists. You just need to change the path upto the name of the repo.


#### Additional information
1. In `test_capture.py`, the `screen_region` variable in `webcam_main` controls where the screen looks at. Feel free to edit this to suit your needs.
2. To quit the program, you can either stop it running from the terminal or press the display box and press q. If you press the 'x' button on the display box, it will just come back.

#### Running
When you are ready, you can run the program via:
```python
python3 testing/test_capture.py
```

### Fade-in Pixels
#### Paths that require changing
Please change the following paths:
- In `testing/mark_detector`, navigate to the `__init__` of the `FaceDetector`. You will see two file paths listed. Change these to match the filepaths on your machine.
- In `testing/test_face_fade`, at the top of the code and in the `__main__` function, you will see two file paths listed. Change these to match the filepaths on your machine.
For both of these, everything after the repo name already exists. You just need to change the path upto the name of the repo.
- In `testing/test_face_fade`, on line 31, change the path to point to an image that exists on your local machine.


#### Running
When you are ready, you can run the program via:
```python
python3 testing/test_face_fade.py
```
