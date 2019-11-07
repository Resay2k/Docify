## 1. Objective

Build a web application that converts handwritten notes to Google Documents.


## 2. How

Tesseract is an optical character recognition (OCR) engine that works very well under closed conditions. To recognize text, there must be a very clean segmentation of the foreground text from the background<sup>[1](#1)</sup>. These conditions are unlikely to be met for the photos uploaded to this app. Therefore, we will employ OpenCV for the preprocessing of the text. We will use it to identify and isolate words, then we will recognize those word through Tesseract.

This python tutorial will initially be used as reference: [https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/](https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/)

Tensor flow can then be used to utilize some of the existing ml handwriting datasets like the IAM dataset.


## 3. Steps

First we will use openCV’s EAST detector to detect text. To do this we follow the instructions provided from [https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/](https://www.pyimagesearch.com/2018/08/20/opencv-text-detection-east-text-detector/)


## 4. Notes


### 4.1 Image Representation

When an image is read into variable using the OpenCV library, it is stored in a numpy array. The array is a 2D matrix with each element representing an image pixel. Aside from height and width, these arrays also have 3rd dimension that represents the number of channels it has:



1. Gray scale images are held in single channel arrays with pixel values ranging from 0-255. The channel holds the grayness of a pixel. 0 representing the color white and 255 representing black.
2.  RGB images are held in 3 channel arrays with pixel values ranging from 255*255*255. The channels represent how red, green and blue a pixel is, respectively.
3. RGBA images are held in 4 channel arrays ranging from 255*255*255*255 with the last channel representing how transparent an image is. 0 represents complete transparency and 255 complete opaqueness.


### 4.2 Deep Learning

“Deep Learning is a fancy term for a Neural Network with many hidden layers. A Neural Network is basically a mathematical model, inspired by the human brain, that can be used to discover patterns in data. The input (your data, for example a spoken sentence "I like cats") goes into the Neural Network, gets processed through the nodes in the hidden layer(s) and output comes out (for example, a prediction for the sounds/words the model estimates were said before, based on the earlier data it has seen). The difference between a Neural Network and Deep Learning is simply that most Neural Networks only have a couple hidden layers, and in Deep Learning there are many more, allowing for more complex patterns.”<sup>[2](#2)</sup>



#### 4.2.1 Layers and Nodes

_“Taking an image from [here](http://cs231n.github.io/neural-networks-1/) will help make this clear._

![images/Text-Recognition0.jpg "image_tooltip"](http://cs231n.github.io/assets/nn1/neural_net2.jpeg)

_Layer is a general term that applies to a collection of 'nodes' operating together at a specific depth within a neural network._

_The **input layer** contains your raw data (you can think of each variable as a 'node')._

_The **hidden layer(s)** are where the black magic happens in neural networks. Each layer is trying to learn different aspects about the data by minimizing an error/cost function. The most intuitive way to understand these layers is in the context of 'image recognition' such as a face. The first layer may learn edge detection, the second may detect eyes, third a nose, etc. This is not exactly what is happening but the idea is to break the problem up into components that different levels of abstraction can piece together much like our own brains work (hence the name 'neural networks')._

_The **output layer** is the simplest, usually consisting of a single output for classification problems. Although it is a single 'node' it is still considered a layer in a neural network as it could contain multiple nodes.”_<sup>[3](#3)</sup>



##

<a name="1">1</a>:
     [https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/](https://www.pyimagesearch.com/2017/07/10/using-tesseract-ocr-python/)

<a name="2">2</a>:[https://www.reddit.com/r/explainlikeimfive/comments/2psgpp/eli5_what_is_deep_learning_and_how_does_it_work/cmzmkr4?utm_source=share&utm_medium=web2x](https://www.reddit.com/r/explainlikeimfive/comments/2psgpp/eli5_what_is_deep_learning_and_how_does_it_work/cmzmkr4?utm_source=share&utm_medium=web2x)

<a name="3">3</a>:
     [https://stackoverflow.com/a/35347548](https://stackoverflow.com/a/35347548)
