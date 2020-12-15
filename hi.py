#This is the DSP lab project done by Anshi,Megha and me
import time
import uuid
import cv2
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelBinarizer

#/Users/mvr/Documents/DSP Project SLD/sign_mnist_test.csv

test_df = pd.read_csv("/Users/mvr/Documents/DSP Project SLD/sign_mnist_test.csv")
train_df = pd.read_csv("/Users/mvr/Documents/DSP Project SLD/sign_mnist_train.csv")
trainLabel=train_df['label']
testLabel=test_df['label']

#NOw encoding the labels 
labelEncoding = LabelBinarizer()
TestLabelEnc = labelEncoding.fit_transform(testLabel)
TrainLabelEnc = labelEncoding.fit_transform(trainLabel)

imgsTrain = train_df.values
imgsTest = test_df.values

x_train = imgsTrain / 255
x_test = imgsTest / 255


x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

#As we have limited data we increase the dataset by rotating and scalling
#we will be using image generator for this purpose


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)