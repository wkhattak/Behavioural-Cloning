# Behavioral Cloning

## Overview   
   
This writeup reflects upon the **Behavioral Cloning** project by explaining how the *steering angle prediction* pipeline works, identifying any shortcomings and proposing potential improvements. 


## Project Goals

The goals/steps of this project are:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Reflection

### Pipeline Overview
1. Load/parse the driving data csv file generated while recording driving a car inside a [simulator](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4419_windows-sim/windows-sim.zip)
2. Explore the input data
3. Balance the input data
4. Design a [Keras](https://keras.io/) CNN, train and validate a CNN regression model
5. Run the simulator in autonomous mode so that the model predicts the steering angles and keeps the car on the road

## Rubric Points

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* **model.py** containing the script to load data, create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing a trained convolution neural network
* **autonomous-run.mp4** showing how the car ran inside the simulator based on the predicted steering angles by the model
* **writeup.md** summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my `drive.py` file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `model.py` file contains the code for loading the csv file, pre-processing the data, training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based on [nVidia](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture and consists of 5 convolution layers and 4 fully connected layers.

The relevant code lines in the `model.py` are [lines 162-224](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L162-L224). For a visual representation, see below section titled *Final Model Architecture*.

#### 2. Attempts to reduce overfitting in the model

As a first attempt to reduce overfitting, *dropout* with 0.75 *keep probability* was tried. However, this resulted in the car hitting the curb and deviating off the road. As a result, *L2 Regularization* with a value of 0.001 was used (model.py [lines 174-195](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L174-L195)). 

The model was trained and validated on different datasets to ensure that the model was not overfitting. This was done by splitting the dataset, such that 80% is used for training and 20% for validation (model.py [line 238](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L238)). The model was tested by running it through the simulator for couple of laps and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an *Adam* optimizer. A *learning rate* of 0.001 was used. Also, the learning rate was decayed by specifying a value of 0.0001 for the *decay* parameter (model.py [line 200](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L200)).

#### 4. Appropriate training data

The training data provided by Udacity was initially chosen. Upon testing, this data alone was not proving to be good enough. So later on, to keep the vehicle driving on the road, further data was recorded using the simulator. I used a combination of center lane driving, recovering from the left and right sides of the road. Also, as the default driving direction is counter-clockwise, few laps were recorded in the clockwise direction in a bid to remove any bias towards left steering angles.

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a very simple CNN comprising one or two convolution layers followed with a similar number of fully connected layers. This is because in the start I wanted to verify that the image processing pipeline is working correctly. Once this was confirmed, then I moved on to making the architecture complex by implementing [nVidia's](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) architecture mainly due to the reason that empirically it seems to be the best model used for solving the exact same problem.

However, even with nVidia's architecture, the results were not promising. Research pointed towards non-uniform dataset to be one of the main reasons. Consequently, I visualized the steering angle distribution (model.py [line 50](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L50))and found out that majority of the data was centered around *zero* angle range as shown in the below histogram:

![Non-uniform histogram](/report-images/histogram-1.png)

The blue line in the histogram shows how the histogram would have looked like if all angle ranges had equal number of samples. As majority of the data is centered towards zero, most of the predictions were being made in this range leading to car hitting the curb on corners.

To rectify this issue, angle ranges with more than the average representation were penalized based on how many samples are in that range;the higher the count, the lesser the probability of inclusion in the training dataset (model.py [line 97](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L97)). 

To further decrease model bias towards taking sharp turns, extreme angles with values -1 & +1 were also removed (model.py [line 81](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L81)).

Another strategy as part of only including relevant data was to remove zero speed images as such images depict scenarios where the car is not actually not moving.

By employing the aforementioned strategies, the distribution of the training dataset was although not uniform, there was a noticable reduction in the number of near-zero angle images as shown below:

![Uniform histogram](/report-images/histogram-2.png)

All this helped to feed in only relevant data, thereby decreasing the training time & reducing the loss

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a continuously decreasing mean squared error on the training set but the mean squared error on the validation set was at first decreasing but then started to increase. This implied that the model was overfitting. 

To combat the overfitting, I modified the model by first introducing *dropout* with a probability of 0.75. However, this resulted in not so optimal driving behavior. Instead I used *L2 regularization* with a value of 0.001 and *decaying learning rate* with a value of 0.0001. The number of epochs was also constrained to five to alleviate overfitting. This resulted in the following loss graph:

![MSE](/report-images/training-mse.png)

The final step was to run the simulator to see how well the car was driving around track one. At firs the car was having issues on the curves and especially at the bridge. To improve the driving behavior in these cases, I recorded recovery data on curves and the bridge. Further to this, I also recorded few laps of track one going clockwise.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py [lines 162-224](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L162-L224)) consisted of a convolution neural network with the following layers:

![Model Architecture](/report-images/network-architecture.jpg)

#### 3. Creation of the Training Set & Training Process

By default, the car drives counter-clockwise on track one is, which produces training data with steering angles mostly in the negative range i.e. towards left. To combat this, few laps of center lane driving were recorded going clockwise. Here is an example image of driving on track one going clockwise:

![Clockwise-drive](/report-images/clockwise.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover back to center line in case it starts to deviate away towards either edges of the road. These images show what a recovery looks like starting from right edge:

![Clcokwise-drive](/report-images/recovery-1.jpg)![Clcokwise-drive](/report-images/recovery-2.jpg)

These images show what a recovery looks like starting from left edge:

![Clcokwise-drive](/report-images/recovery-3.jpg)![Clcokwise-drive](/report-images/recovery-4.jpg)

As I had recorded few laps going clockwise, I didn't feel the need to flip images. 

After the collection process, I had 58,251 number of images. However, after balancing the dataset (model.py [line 71](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L71)), as discussed above, I was left with 19,417 images.

![Dataset-count](/report-images/dataset-count.png) 

But this was not the final number of images used for training as I also used the left and right camera images by adding & subtracting a correction factor of 0.20 from left & right steering angles respectively(model.py [lines 126-130](https://github.com/wkhattak/Behavioural-Cloning/blob/master/model.py#L126-L130)). This resulted in tripling the amount of training dataset. Before feeding the images to the model, each image was resized to 66 x 200 as required by the nVidia architecture.

I finally randomly shuffled the dataset and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as going any higher was not decreasing the validation loss. 

### Shortcomings/Future Enhancements

* Only the left track was used for data collection & training. In future, the right track (challenge track) can also be used to record driving behavior data & further generalize the model.

* Image jittering techniques such as *horizon shift*, *brightness adjustment* can further be introduced to reduce overfitting.

## References

* https://stackoverflow.com/questions/11349333/when-processing-csv-data-how-do-i-ignore-the-first-line-of-data

* https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/

* https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

* https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project

* https://github.com/darienmt/CarND-Behavioral-Cloning-P3

* https://github.com/naokishibuya/car-behavioral-cloning

* https://sonamsingh19.github.io/blog/2016/09/01/ELU-For-RNN.html

* http://forums.fast.ai/t/what-is-subsample-of-convolution2d-doing/3555

* https://www.quora.com/How-do-we-detect-overfitting-and-under-fitting-in-Machine-Learning
