# **Project 2: Traffic Sign Recognition** 


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

[image1]: ./images/image1.png 
[image2]: ./images/image2.png 
[image3]: ./images/image3.png 
[image4]: ./images/image4.png 
[image5]: ./images/image5.png 
[image6]: ./images/image6.png 
[image7]: ./images/image7.png 
[image8]: ./images/image8.png 
[image9]: ./images/image9.png
[image10]: ./images/image10.png
[image11]: ./images/image11.png

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here are links to my [Ipython notebook with code](https://github.com/lingyun-wu/CarND-Project-02/blob/master/Traffic_Sign_Classifier.ipynb) and the [HTML output](https://github.com/lingyun-wu/CarND-Project-02/blob/master/report.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here are two images which give us an exploratory visulization of the dataset.

The first image shows an sample graph for each traffic sign label.

![alt text][image2]

The second image is a horizontal bar chart showing the size of class in the dataset. It can be seen that the data are not evenly distributed among classes. This problem would cause bia in the classifier towards the classes with larger trainning sample sizes. It can be solved by data augmentation in the second part.

![alt text][image1]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

**1.1** In the preprocessing process, first, I used the Contrast-limited adaptive histogram equalization (CLAHE) method to enhance the contrast of each image. This is because some of the images are taken under very dark environment, the CLAHE process can help improve the contrast of the images and make them brighter. Below are some sample images.

![alt text][image3]

After CLAHE, I normalized each image so it has a uniform scale distributed between [-1, 1], which is better for the model to learn. 


**1.2** From the bar chart of the data set above it can be seen that the numbers of different classes are not evenly distributed, so I randomly implement certain image transformation techniques to augment the data set. The techniques used for generating additional images are translation, rotation, affine tranformation, and Gaussian smooting.
Here are some graphs showing the transformation results.
![alt text][image4]
 
The difference between the original data set and the augmented data set can be showed from the bar chart below.
![alt text][image5]
![alt text][image6] 



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I tried two architectures in this project. One is the LeNet architecture, and the other one is from the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks" which is written by Lecun et al.
The image below is the architecture I used as the final model.
![alt text][image7]

This model consisted of the following layers:

|   #   | Layer         		|     Input                    |   Output   |
|:-----:|:-----------------------------:|:----------------------------:|:----------:| 
| 1     | Conv   		        | 32x32x3                      |  28x28x6   | 
|      	| Relu                          |                              |            |
|       | Max pooling	      	        | 28x28x6                      | 14x14x6    |
| 2     | Conv                          | 14x14x6	               | 10x10x16   |
|       | Relu          		|         	               |            |
|       | Max Pooling	 		| 10x10x16	               | 5x5x16     |
| 3     | Conv                          | 5x5x16                       | 1x1x294    |
|	| Flatten			| 1x1x294                      | 294	    |
|Branched| Max Pooling                  | Results of 1st pooling       | 7x7x6      |
|       | Flatten                       | 7x7x6                        | 294        |
|       | Concatation                   | Results of two flatten       | 588        |
| 4     | Fully Connected               | 588                          | 120        |
|       | Relu                          |                              |            |
|       | Dropout                       |                              |            |
| 5     | Fully Connected               | 120                          |  43        |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The optimizer I used is "AdamOptimizer";
The batch size is 128;
The number of epochs is 100;
The learning rate is 0.001;


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* The average validation set accuracy after 50th epoch is 0.967
* The test set accuracy is 0.955

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture was LeNet. It was chosen because it was easy to implement.

* What were some problems with the initial architecture?
The accuracy was not high enough.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I changed the architecture to the one showed in the table above. I also added the dropout process in the 4th layer in order to prevent overfitting.



If a well known architecture was chosen:
* What architecture was chosen?
I chose the architecture described by Lecun in the paper "Traffic Sign Recognition with Multi-Scale Convolutional Networks".

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
The validation data set has an accuracy of 0.967, and the test data set has an accuracy of 0.9555. Because the model hasn't "seen" the test data set before and its accuracy of 0.955 is not that bad. It seems like the model works OK.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][image8]

All of the images are very easy to identify by human eyes.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry      		| No entry   									| 
| No vehicles     			| No vehicles 										|
| Yiled					| Yield											|
| 30 km/h	      		| 30 km/h					 				|
| 70 km/h			| 70 km/h      							|
| Road work                     | 80 km/h        |

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83.33%. The model was doing well for the first 5 images, which is not so surprising since all of them are very clear to see. But for the last one "road work" image, the model predicted it as the speed limit sign of 80 km/h. This might be the reason that all the training data set of the "road work" sign are not so clear as the one here, which makes the model have a hard time to identy a clear "road work" sign..

#### 3. Describe how certain the model is when predicting on each of the six new images by looking at the softmax probabilities for each prediction. Provide the top 3 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Here are the bar charts showing the softmax probablities of predicting on each of the six new images by the model.

It seems like that the model is pretty sure that the last image is speed limit (80km/h), but it turns out that it is "road work". It has a really hard time in identifying the "road work" sign.

![alt text][image9]


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here are the orignal sample image and the one after pre-processing.

![alt text][image10]

Here are the images output of trained network's fearture maps.

![alt text][image11]

From the pictures above, it seems like that the neural network tried to identify different parts of the image in order to make classification of it. 
