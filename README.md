
# Team 11008, ParyavaranAI
## MeghNA - Megh Nowcasting and Analytics

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

***Problem Statement:***
Develop and implement an algorithm to:
- Detect clouds in INSAT satellite images and
- Predict the location of clouds in subsequent images.

***Solution***
An interactive web tool that allows users to interact with our cloud analytics engine
offering:
1. **Cloud detection**
- Using clustering and feature thresholds - **KMeans Clustering**
- Using Neural Networks - **Mask RCNN** 
2. **Nowcasting: Cloud motion prediction**
- Using modified Mean Path Adjustment **MPA**
- Using Neural Networks - **CNN + LSTM**
3. **Cloud classification**
- Using infrared v/s visible image membership
4. **Cloud attributes**
- Based on cloud type, TIR1 and VIS count over infrared and visible satellite images.

***Dataset Study***
- INSAT-3D captures through thermal infrared
and visible waves channels are provided.
- Images are captured every 30 minutes.

## Folder Walkthrough
| Folder Name | Service |
|--|--|
| [CNN_LSTM](https://github.com/manandoshi1607/NM373_ParyavaranAI/tree/master/CNN_LSTM) |  Code for cloud motion prediction using CNN+LSTM
| [Mask_RCNN](https://github.com/manandoshi1607/NM373_ParyavaranAI/tree/master/Mask%20RCNN) | Code for cloud detection using Mask RCNN
| [classification](https://github.com/manandoshi1607/NM373_ParyavaranAI/tree/master/classification) | Code for cloud classification types

## Technology Used

- OpenCV
- Tensorflow
- Keras
- Angular
- Django Rest Framework
- SQLite
- Gdal and Rasterio

## Cloud Detection
### K-Means Clustering
- Calculate feature vector for each pixel
- Obtain Cloud Mask after clustering pixels
- Apply edge filter to mark cloud edges
- Using flood fill to label individual clouds

#### Results:

##### 1. Original Satellite Image

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/Original_Capture.png?raw=True)

##### 2. Clouds Mask

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/Cloud_Mask.png?raw=True)

##### 3. Cloud Edges Marked

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/Edges_Marked.png?raw=True)

##### KMeans Result - Labelled Clouds

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/Clouds_Labelled.png?raw=True)

### Mask RCNN

- Manual Annotation of images
- From all the images given, three major cloud portions were identified
- On training and testing the Mask RCNN model on images from the visible channel it was understood that the cloud regions are not uniform due to the daylight that can be eventually seen in the images.
- On training and testing the Mask RCNN model on images from the  infrared channel it was found that there are three major cloud regions
- The model is fine tuned on the coco weights.
 - It is run for 10 epochs with a batch size of 75.

#### Results:

##### Satellite Image

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/satellite1.jpg?raw=True)

##### Mask RCNN Result

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/satellite_1.png?raw=True)

##### Model H5 Link
[Mask RCNN](https://drive.google.com/file/d/1-o9oSBf6FhlEkzMUsboSE5ce4HQ0tIyS/view?usp=sharing)

## Cloud Motion Prediction

### Mean Path Adjustment
 - Center of Mass is traced to predict cloud movements.
 - Cloud features are compared to match clouds in next images
 - Steps involved 
	 -  Let previous CoM positions be: t-1, t-2
   -  Let at be the actual CoM position at t
   	- Predicted mean CoM at t = mean(t-1, t-2)
   	- Next prediction for t+1:
	   	- t+1 = mean(at, t) + (at-t)

### CNN+LSTM
- Sequence Size - 4 images
- Model Architecture:
	-   BatchNormalization
	-   3 LFLBs (2dConv + BN + 2dConv + BN + MaxPool + Dropout)
	-   Flatten
	-   2 LSTMS (n_units = 512)
	-   Dense (300 * 300)
- Model trained for Epochs: 100
-   Training Data:
    -   X: Image 0 - Image 3
    -   Y: Image 4
-   Validation Data:
    -   X: Image 4 - Image 7
    -   Y: Image 8
-   Test Data:
    -   X: Image 8 - Image 11
    -   Y: Image 12
- Model Outputs:
	-   **_loss: 55.8962 - val_loss: 339.6556_**
	-   Time taken to train:  **_174.56293845176697 seconds_**
	-   Time taken to generate output:  **_0.13999462127685547 seconds_**
 
#### Model Architecture
![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/motion_architecture.jpeg?raw=True)

#### Results:

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/motion_2.jpeg?raw=True)

![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/motion_1.jpeg?raw=True)


##### Model H5 Link
[CNN LSTM](https://drive.google.com/file/d/1Th9ikwmyTGipz6Z8sfTj4QIsPR5whtOO/view?usp=sharing)

## Cloud Classification
- Obtained Mapping between intensity values and thermal infrared.
- From various sources we have set thresholds for determining cloud height. 
	-	Cyclone region : <200K
	-	High clouds : 200 - 243K
	-	Middle clouds : 243 -270 K
	-	Low clouds : >270K

We take the given mask, convert it to corresponding temperature values and then classify each pixel. The overall mask classification is based on the category that the maximum number of pixels fall in.

#### Results:
![Image Description](https://github.com/manandoshi1607/NM373_ParyavaranAI/blob/master/docs/classification.jpeg?raw=True)

## References 
[Links to paper referred](https://drive.google.com/file/d/17kpehylbqtQWQpIVDHA670K9nCZei4Ul/view)

### Phase I PPT
[Phase I PPT](https://drive.google.com/file/d/1NKji3qKxTO0GgxOqGBMyX8V147v2BIXM/view?usp=sharing)

