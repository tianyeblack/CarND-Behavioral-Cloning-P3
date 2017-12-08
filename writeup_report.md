# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./center_2017_12_04_21_31_18_849.jpg "Normal Image"
[image3]: ./vf_center_2017_12_04_21_31_18_849.jpg "Flipped Image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* transfer_learning_model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```python
python drive.py transfer_learning_model_epoch6.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network very similar to what's described in one of Nvidia's paper (arXiv:1704.07911v1). The main part is:
1. First, 3 convolution layers with 5x5 filter size, 2x2 stride, depths of 24, 36 and 48, valid padding and rectified linear activation (model.py lines 43-45)
2. Then followed by 2 convolution layers with 3x3 filter size, 1x1 stride, depths of 64 and 64, valid padding and rectified linear activation (model.py lines 46-47)
3. Then 3 fully connected layers with 100, 50 and 10 neurons each.

The model includes RELU activations to introduce nonlinearity, and the data is normalized in the model using a Keras Lambda layer (model.py line 41). A cropping layer is introduced to remove noisy background features so the model focuses on the lanes and road surface (model.py line 42).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 49). Otherwise, the model overfits my data around the 3rd epoch.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 35-37). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track and mostly center of the lane.

The data is also pre-augmented before loading into memory with generator. This is to achieve more accurate randomness. If the center image is always loaded together with the left and right ones or the original is always loaded with its flipped one, bias may be introduced and the model would overfit that bias. However, the result does not differ much here probably because the situation is not too complicated.

#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and right camera images with hardcoded correction and flipped images of center, left and right.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with existing CNNs and then tailor it to my needs.

My first step was to use a convolution neural network model similar to the one Nvidia proposed (arXiv:1704.07911v1). I thought this model might be appropriate because it is proven working in real world environment and with some tuning it can learn the features of the simulated environment.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80 and 20 percent each). It does not fit very well, with validation loss around 0.2 and testing with simulator only slowly but consistently led the car off the track without turning.

I tried to simplify the model, which somehow improved the behaviors a little but still crashed out of the track at certain turns. I also compared the provided data set, which has recovery laps. With that the model had problems driving on the bridge and only slightly improved at turns at low speeds (9 or 15mph).

Then I checked my data pipeline again and finally realized I wasn't training on the "right" data. One peculiarity of cv2 is the image read in is BGR, which is the opposite of RGB. I didn't grayscale the data thinking the model needs to recognize the difference in colors. However, my way of converting it from BGR to RGB was wrong. I put it in model as one of the Lambda layers and that applied to both training and simulation images.

After correcting that, the loss dropped but overfit very soon, at around 3 epochs. I added a Dropout layer with rate of 0.5 after the flattening layer. Then the loss dropped even more to around 0.01 and the car followed the lane in the center very well.

With some trial and error, I found the model reaches the lowest training error at around epoch 6.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road and remain mostly at the center of the lane at full speed (30mph).

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:
1. First, 3 convolution layers with 5x5 filter size, 2x2 stride, depths of 24, 36 and 48, valid padding and rectified linear activation (model.py lines 43-45)
2. Then followed by 2 convolution layers with 3x3 filter size, 1x1 stride, depths of 64 and 64, valid padding and rectified linear activation (model.py lines 46-47)
3. Then 3 fully connected layers with 100, 50 and 10 neurons each.

Here is a visualization of the architecture

![Model Visualization][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 4 laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving Counter clockwise][image2]

Then I repeated this process on track one but a different direction.

After the collection process, I had 10238 number of data points. I then preprocessed the csv by splitting out center, left and right images to each of its own line and add hardcoded correction to the steering angles.

To further augment the data set, I also flipped all images and angles with the newly generated csv thinking that this would double the data set size and still match actual driving since the behaviors for lane following are symmetrical. For example, here is an image and its flipped one:

![Original Image][image2]
![Flipped Image][image3]

I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 as evidenced by loss visualization. After the 6th epoch validation loss stops to improve as much as training loss but not too much to over fit the data a lot. I used an Adam optimizer so that manually training the learning rate wasn't necessary.
