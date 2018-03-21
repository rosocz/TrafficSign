# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: images/samples_distribution.png "Distribution of samples"
[image2]: examples/grayscale.jpg "Grayscaling"
[image3]: examples/random_noise.jpg "Random Noise"
[image4]: external/120_32.jpg "Traffic Sign 1"
[image5]: external/end_32.jpg "Traffic Sign 2"
[image6]: external/schule_32.jpg "Traffic Sign 3"
[image7]: external/stop_32.jpg "Traffic Sign 4"
[image8]: external/wild_32.jpg "Traffic Sign 5"
[image9]: images/rgb.png "RGB"
[image10]: images/gray.png "Gray"
[image11]: images/30.png "Original"
[image12]: images/30_augumented.png "Augumented"
[image13]: images/distribution_augumented.png "Distribution after augumentation"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rosocz/TrafficSign/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the training data are distributed among classes. It's for me one of the key findings and great hint how to act with data.

![alt text][image1]
![alt text][image13]

Before and after augumentation.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it reduces depth of image from 3 to 1, all succesive transformations work with less data and job to be done is easier.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image9]
![alt text][image10]

Next I tried histogram equalization with function cumulative distribution, it improves contrast of edges.
I tried to remove noise with median filter. Many images are taken with bad light condition.
Standardization centers data to have zero mean, normalization data by using minmax method puts data in range of <-1,1>.

I decided to generate additional data because the number of sample images is very low. Some classes have around 60 images where many of them has bad quality.
Dataset has significant disproportion, from 60 to 750 samples of single class.

To add more data to the the data set, I used the following techniques:

* rotation
Image rotation from -15 to 15 degrees, random select. Not all signs are perfectly vertical, rotating them generates useful additional images.
* sharpen
Images after sharpening contain much significant edges and their detection is easier
* shrink
Size of signs on images varies, this transformation creates more images simulating these cases.
* add noise
Adding noise creates next group of images, random noise creates new unique images.

I tried to equalize number of sample images per class therefore I'm taking maximum number of sample images for a single class in training dataset, counting +50% 
and creating augument images so each class has more or less same number of sample images. Up to 2 transformation can be applied.  I hope that I have enough transformation combinations to generate valid group of new images.


Here is an example of an original image and an augmented image:

![alt text][image11]
![alt text][image12]

The difference between the original data set and the augmented data set is the following ... 

All classes have equalized number of sample images and total count of samples is much higher. As I wrote above, it's random combination of rotation, sharpening edges, shrinking and adding some noise.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

As a first model I tried same model as was used for LeNet with valid padding and max depth 16. I think I'm loosing too much with valid padding, same padding is usefull because I don't loose data.
This model gave me about 91% validation accuracy. After reading few hints how to improve training of NN, I changed model to have higher depth and the accuracy starts to grow, somewhere to 93%.
Decreasing learning rate and batch size gave me final solution.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 gray image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x32 				|
| Convolution 5x5     	| 1x1 stride, same padding, outputs 16x16x64 	|
| RELU					|												|
| Max pooling	      	| 4x4 stride,  outputs 4x4x64 				|
| Flatten	      	| outputs 1024 				|
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout		|         									|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Fully connected		| outputs 43        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used smaller size of batch, 64 and learning rate 0.0005. Decreased values increase learning time, but results are better. I set number of epochs to 100, it gives me quite good accuracy, around 95%. I tried 
130 epochs as well, I got better results but the time to train NN was too long. So there is a place to grow if required.
I choosed Adam optimizer, it's taken from LeNet solution and I guess it's good enough, generally speaking, it has better result than stochastic gradient descent. Hyperparameters were choosen only by guessing the best combination.
Mean is 0, standard deviation 0.1 and dropout probability 0.3


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.894
* validation set accuracy of 0.95 
* test set accuracy of 0.929

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
First solution was taken from LeNet. Valid padding of 2 convolutional networks. 3 fully connected networks, with RELU and maxpool. It was recommended in the course.
* What were some problems with the initial architecture?
I was facing problem with accuracy, the model was reaching 91% only.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I was quite happy with the basic design of 2 covnets and 3 fully connected, I tried to add more depth, dropout to fully connected layer, I changed padding from valid to same, my idea was to prevent loosing data by padding.
* Which parameters were tuned? How were they adjusted and why?
Learning rate and batch size were decreased, it is general recommendation to decrease it to get better results. As I was able run code on AWS, I used 100 epochs and result were just fine.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I think most important is learning rate, batch size and number of epochs. Convolution layer breaks up image to smaller parts, so it can identify all specific shapes related to each sign 
and by using whole group of sample images it finds connection between similar shapes and label. Adding dropout was not so big impact as I have seen. Removing random nodes could be more usefull for much larger neural neworks. 

If a well known architecture was chosen:
* What architecture was chosen?
I used architecture of LeNet
* Why did you believe it would be relevant to the traffic sign application?
I worked well for character, so it could work for traffic signs as well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
It doesnt seems to be overfitting or underfitting, accuracy level is acceptable.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because its a bit skew and some training sessions omit number "1" and classified it as 20km/h. Constant problem is with third image, where it's very often classified as pedestrians. Classification of this image is not very stable.
5th image, wild animals crossing, had low accuracy level as well, after additional augumentation activity is quite stable.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| Children crossing     			| Pedestrians 										|
| 120km/h					| 120km/h											|
| Stop	      		| Stop					 				|
| Wild animals crossing			| Wild animals crossing      							|


The model was able to correctly guess all 5 traffic signs, which gives an accuracy of 100%. But as I mentioned, 3rd classification is not stable. This compares favorably to the accuracy on the test set of 100%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image, the model is sure that this is a No entry sign (probability of 0.99), and the image does contain a No entry sign. All other probabilities are insignificant. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9987         			| No entry sign   									| 
| 0.0012     				| Speed limit (100km/h) 										|
| 0.00005					| Roundabout mandatory											|
| 4.7e-08	      			| No passing for vehicles over 3.5 metric tons					 				|
| 2.8e-08				    | Priority road      							|


Second image contains Children crossing, the model is pretty sure (probability of 0.99), but classification of this image is not stable at all. The result is changing after every run. All other probabilities are insignificant. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9947         			| Children crossing   									| 
| 0.0038     				| Right-of-way at the next intersection 										|
| 0.0005					| Priority road											|
| 0.00048	      			| Slippery road					 				|
| 0.0004				    | End of no passing by vehicles over 3.5 metric tons      							|


Third image contains Speed limit (120km/h), it's classified correctly with probability of 0.99. All other classifications are realted to speed limits only and probabilities are low. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9988         			| Speed limit (120km/h)   									| 
| 0.0011     				| Speed limit (70km/h) 										|
| 2.7e-05					| Speed limit (80km/h)											|
| 1.4e-05	      			| Speed limit (50km/h)					 				|
| 8.4e-06				    | Speed limit (100km/h)      							|

Fourth image contains Stop sign, correctly classified with probability of 0.99. All other probabilities are low. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999         			| Stop   									| 
| 3.0e-05     				| Turn left ahead 										|
| 8.0e-10					| Speed limit (120km/h)											|
| 4.9e-10	      			| Keep right					 				|
| 4.9e-13				    | No vehicles      							|

Last, fifth image contains Stop sign, correctly classified with probability of 0.99. All other probabilities are low. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9999         			| Wild animals crossing   									| 
| 3.0e-05     				| Turn right ahead 										|
| 8.0e-10					| Bicycles crossing											|
| 4.9e-10	      			| No entry					 				|
| 4.9e-13				    | Beware of ice/snow      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


