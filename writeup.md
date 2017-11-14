#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[DistributionBefore]: ./images/DataDistribution.png "Data Distribution Before Augmentation"
[DistributionAfter]: ./images/DataDistributionAfterAugmentation.png "Data Distribution After Augmentation"
[Normalize]: ./images/GrayNormalize.png "Normalization"
[Augmentations]: ./images/Augmentations.png "Traffic Sign Augmentations"
[LeNet]: ./images/MyLeNet.png "LeNet Model"
[GTSRB]: ./images/GtsrbImages.png "GTSRB Images"
[GTSRBPredicted]: ./images/GtsrbPredictions.png "GTSRB Predictions"
[Visualization]: ./images/NetworkVisualization.png "Visualization of Convolution Activation"
[Accuracy]: ./images/ValidationAccuracy.png "Training Validation Accuracy"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/rajatsharma01/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a seaborn distplot with kde line showing high variance in different classes of traffic sign data. Note that the distribution looks similar across training, validation and test set.

![DistributionBefore]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because grayscale images tend to increase accuracy for CNN. Also it is compuntationally less expensive to train with grascale images than RGB. I have used tensorflow's tf.image.rgb_to_grayscale function to convert image to grayscale. To normalize these grayscale images, I have used tensorflow's tf.image.per_image_standardization function. Normalization allows gradient descent to achieve global minimum at a fater rate as largely varying input (pixel intensity in this case) can start off with a biased configuration of favoring or penalizing some features (pixels) over other.

Here is an example of a traffic sign images randomly selected from training data: original, after grayscale conversion and normalization.

![Normalize]

Data distribution among training images varies by large, which is not good for training Nerual Networks as it may learn to identify traffic with more number of samples with more accuracy. Also adding random noise to original images help NN to learn variations of a traffic sign. I have used following transformations for augmenting images:

1. Adjust brightness of images with random factor using tf.image.random_brightness
2. Using projective transformations to rotate, shift and shear images. I have used tensorflow.contrib.image module's compose_transform and transform functions.
3. Randomly flipping some of these images lef-right (tf.image.random_flip_left_right) or up-down (tf.image.random_flip_up_down)

Overall, I tried to use tensorflow during this project to enhance my learning for Tensorflow.

Here is an example of an original image for each class and corresponding augmented images:

![Augmentations]

I augmented the original training dataset with above transformed image to have atleast a minium number of samples (1000 while submitting) for each traffic sign class. And this is how data distribution looks like after augmenting training dataset

![DistributionAfter]


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

Following diagram shows my model implementation of LeNet. Note that its essentially similar to LeNet implimentation done in LeNet lab, with one change though, I am feeding output of First layer subsampling to fully connected Layer ad described by Pierre Sermanet and Yann LeCun's [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). Additional subsampling is done to first layer output to bring subsampling at same level.

![LeNet]
 
I had used VALID sampling with 1x1 stride size for both convolution and pooling. I have used ELU activation instead of RELU to avoid vanishing gradient problem associated with RELU.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used following hyperparameters to train my model
BATCH_SIZE = 256
LEARN_RATE = 0.001
EPOCHS = 50

To assign initial weights, I have used xavier initialization to set weights which are not too large or not too small. Finally, I have used Adam optimizer for calculating gradient descent.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I started with basic implimentation of LeNet without any data augmentations, I was achieving about 0.96 validation accuracy. However very low test accuracy on GRSRB images. Based on suggestions in rubric page, I stared with implementing image augmentation techniques one by one, modified my model to have deeper and more number of hidden layers. However I started hitting limitation of my laptop to train such model as it used to take many hours to train, so I switched back to smaller version of model which is closer to basic LeNet. I also noticed that too many data augmentations were causing validation accuracy to be low, so I decided to stick with 1000 samples per class to generated additional required augmentations. Finally what I achieved seems to be a middle ground solution, not very high accuracy though. I would come back to fix my model to scale well when I have access to GPU based machine.

My final model results were:
* training set accuracy of : Didn't really measure it separately
* validation set accuracy of 0.946
* test set accuracy of 0.933

Following graph shows validation accuracy plot for each epoch

![Accuracy]

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I have downloaded all GTSRB images and I randomly pcik 5 images to test, e.g. these were the images picked in last run:

![GTSRB]

Some of these images are difficult to classify for lower brightness level e.g. images 2, 3 and 4 in above set, all these images are very dark, however my model could classify them correctly.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

In this set, my model could predict all of them in this run, however I have often seen 80% accuracy when testing with larger set of images. Predicted images are marked with a wright vs wrong tick mark, here is the predicted images result:

![GTSRBPredicted]

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 30th cell of the Ipython notebook.

The top 3 soft max probabilities were (output in 32nd cell of my Ipython notebook)

For the first image, there is strong decision about probabilty for correct prediction.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| ClassId: 1 Speed limit (30km/h)  									| 
| 0.39646107     				| ClassId: 32 End of all speed and passing limits										|
| 0.14772479					| ClassId: 31	Wild animals crossing										|

For next 4 images, although model could correctly predict signs, but all top 3 probabilities are very close, so even though result is correct, predictions are on the edge.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999487         			| ClassId: 4 Speed limit (70km/h)  									| 
| 0.94174737     				| ClassId: 0 Speed limit (20km/h)										|
| 0.82010305					| ClassId: 24	Road narrows on the right										|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| ClassId: 7 Speed limit (100km/h)  									| 
| 0.99650842     				| ClassId: 10 No passing for vehicles over 3.5 metric tons										|
| 0.9887383 					| ClassId: 42	End of no passing by vehicles over 3.5 metric tons										|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99999893         			| ClassId: 36 Go straight or right  									| 
| 0.99999774     				| ClassId: 17 No entry										|
| 0.99999511					| ClassId: 28	Children crossing										|

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| ClassId: 2 Speed limit (50km/h)  									| 
| 0.99682069     				| ClassId: 3 Speed limit (60km/h)										|
| 0.99270004					| ClassId: 14	Stop										|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Following is the output of first two convolution layers activation for the first GTSRB image: Speed Limit 30 km/h. First convolution output seems to be learning digits of 30 and circle around it.

![Visualization]
