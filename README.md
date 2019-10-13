# cnn_face_recogntion

![Python 3.7](https://img.shields.io/badge/python-3.7-blue.svg)
![Tensorflow 1.14](https://img.shields.io/badge/tensorflow-1.14-orange.svg)
![Keras 2.2](https://img.shields.io/badge/keras-2.2-red.svg)

A project of face recogntion used Convolution Nerual Network (CNN)

## Data

We have 1000 images. We use 600 images as training data. The rest is in validation data. 

In training data, we crop labeled face images as positive instances. We generate negative instances by cutting out the images located outside the face zone. All these positve and negative instances build up the training set.

We do the same thing for the validation set.

