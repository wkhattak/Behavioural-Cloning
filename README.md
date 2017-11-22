# Project: Behavioral Cloning
## Overview   
   
This writeup reflects upon the **Behavioral Cloning** project by explaining how the *steering angle prediction* pipeline works, identifying any shortcomings and proposing potential improvements. For a detailed description, please see [prohect report](https://github.com/wkhattak/Behavioural-Cloning/blob/master/writeup.md).


## Project Goals

The goals/steps of this project are:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Directory Structure
* **model.py:** Script to load data, create and train the model
* **drive.py:** Script for driving the car in autonomous mode
* **model.h5:** File containing a trained convolution neural network
* **video.mp4:** Video clip showing how the car ran inside the simulator based on the predicted steering angles by the model
* **writeup.md:** Project report
* **writeup-images:** Directory containing images that appear in the project report
* **README.md:** Project readme file

## Requirements
* Python-3.5.2
* OpenCV-3.3.0
* Numpy
* TensorFlow
* Scikit-learn
* Matplotlib
* Keras
* [Udacity Driving Simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)

## Usage/Examples
The `model.py` file contains the code for loading the training data (csv & images), pre-processing the data, training and saving the convolution neural network. The file shows the pipeline used for training and validating the model, and it contains comments to explain how the code works.

Using the Udacity provided simulator and the `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

## Troubleshooting

**libgtk**

The command import cv2 may result in the following error. `ImportError: libgtk-x11-2.0.so.0: cannot open shared object file: No such file or directory.` To address make sure you are switched into the correct environment and try `source activate [your env name];conda install opencv`. If that is unsuccessful please try `apt-get update;apt-get install libgtk2.0-0` (may need to call as sudo).

## License
The content of this project is licensed under the [Creative Commons Attribution 3.0 license](https://creativecommons.org/licenses/by/3.0/us/deed.en_US).