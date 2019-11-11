## 1. Objective

Build a web application that converts handwritten notes to Google Documents.


## 2. How

Tesseract is an optical character recognition (OCR) engine that works very well under closed conditions. To recognize text, there must be a very clean segmentation of the foreground text from the background<sup>[1](#1)</sup>. These conditions are unlikely to be met for the photos uploaded to this app. Therefore, we will employ OpenCV for the preprocessing of the text. We will use it to identify and isolate characters, then we will read those characters through Tesseract.

This python tutorial will initially be used as reference: [https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/](https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/)

A TensorFlow implementation can then be made so  we can make use of the existing ml handwriting datasets like the IAM dataset

## 3. Steps

First we will use openCV’s EAST detector to detect text. To do this we follow the instructions provided from [https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)


## 4. Notes


### 4.1 Image Representation

When an image is read into variable using the OpenCV library, it is stored in a numpy array. The array is a 2D matrix with each element representing an image pixel. Aside from height and width, these arrays also have 3rd dimension that represents the number of ‘channels’ it has:



1. Gray scale images are held in single channel arrays with pixel values ranging from 0-255. The channel holds the ‘grayness of a pixel’. 0 representing white and 255 representing.
2.  RGB images are held in 3 channel arrays with pixel values ranging from 255*255*255. The channels represent how red, green or blue a pixel is.
3. RGBA images are held in 4 channel arrays ranging from 255*255*255*255 with the last channel representing how transparent an image is. 0 representing complete transparency and 255 complete opaqueness.

#### 4.1.1 Mean Subtraction

The image is preprocessed into a blob before being inputted into the EAST detector. One of the additional operations conducted in the process is mean subtraction; the mean intensity values for the rgb channels are subtracted from the image. This centers the data values around zero and helps train the neural net faster as it can ignore the lighting variations of different images. For the EAST detector, the average mean intensity values it used while being trained where 123.68, 116.78, 103.94.



### 4.2 Deep Learning

“Deep Learning is a fancy term for a Neural Network with many hidden layers. A Neural Network is basically a mathematical model, inspired by the human brain, that can be used to discover patterns in data. The input (your data, for example a spoken sentence "I like cats") goes into the Neural Network, gets processed through the nodes in the hidden layer(s) and output comes out (for example, a prediction for the sounds/words the model estimates were said before, based on the earlier data it has seen). The difference between a Neural Network and Deep Learning is simply that most Neural Networks only have a couple hidden layers, and in Deep Learning there are many more, allowing for more complex patterns.”<sup>[2](#2)</sup>



#### 4.2.1 Layers and Nodes

_“Taking an image from [here](http://cs231n.github.io/neural-networks-1/) will help make this clear._

![images/Text-Recognition0.jpg "image_tooltip"](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)

_Layer is a general term that applies to a collection of 'nodes' operating together at a specific depth within a neural network._

_The **input layer** contains your raw data (you can think of each variable as a 'node')._

_The **hidden layer(s)** are where the black magic happens in neural networks. Each layer is trying to learn different aspects about the data by minimizing an error/cost function. The most intuitive way to understand these layers is in the context of 'image recognition' such as a face. The first layer may learn edge detection, the second may detect eyes, third a nose, etc. This is not exactly what is happening but the idea is to break the problem up into components that different levels of abstraction can piece together much like our own brains work (hence the name 'neural networks')._

_The **output layer** is the simplest, usually consisting of a single output for classification problems. Although it is a single 'node' it is still considered a layer in a neural network as it could contain multiple nodes.”_<sup>[3](#3)</sup>

#### 4.2.2 Forward/Back propagation

Forward propagation is the term used when the input is passed through the neural network and an output is calculated as a result.[^4] It’s pretty straightforward. Back propagation occurs during the learning process of a neural net where the gradient of the output error is used to go back and determine the weights of the layers in the network. It is the mechanism by which neural networks learn.


### 4.3 East Detector Output

After the image is passed through the net, we get two outputs; a map of informing us the likelihood of a specific region containing text, and another map that holds the bounds for that region. The first is called a **scores map**, the second is called the **geometry map.** The score map values for each pixel range from 0 to 1. The score stands for the confidence of the geometry shape predicted at the same location. Think of the geometry map to be something like the following figure taken from [here](https://medium.com/@andersasac/anchor-boxes-the-key-to-quality-object-detection-ddf9d612d4f9)

![](https://miro.medium.com/max/1200/1*q7bYvvpZWeZ9dBVB4yI8EA.png)

### 4.4 Text Detection Boxes

To generate bounding boxes on the images:



1.  We first get the scores and their corresponding geometry maps
2.  Iterate through each row and get the scores and box coordinates in it
3.  Then iterate through each column. We ignore boxes with insufficient scores and collect the following
1. scale the coordinates by 4 because the EAST detector shrinks volume size of     original image by 4x after processing
2. get rotation angle of the bounded box along with cosine and sine values
3. get height and width of bounded box then convert to (x,y)-coordinates on image. Finally add them to an array of bounding boxes and scores.
4. Apply non-maxima suppression to suppress weak, overlapping bounding boxes. Something like this, found [here]( https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
![](https://miro.medium.com/max/3116/1*6d_D0ySg-kOvfrzIRwHIiA.png)

5. Draw the bounding box on the image


## Notes

<a name="1">1</a>:
     [https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)

<a name="2">2</a>:[https://www.reddit.com/r/explainlikeimfive/comments/2psgpp/eli5_what_is_deep_learning_and_how_does_it_work/cmzmkr4?utm_source=share&utm_medium=web2x](https://www.reddit.com/r/explainlikeimfive/comments/2psgpp/eli5_what_is_deep_learning_and_how_does_it_work/cmzmkr4?utm_source=share&utm_medium=web2x)

<a name="3">3</a>:
     [https://stackoverflow.com/a/35347548](https://stackoverflow.com/a/35347548) 
