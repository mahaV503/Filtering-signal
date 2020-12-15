#This is the DSP lab project done by Anshi,Megha and me
import time
import uuid
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
#/Users/mvr/Documents/DSP Project SLD/sign_mnist_test.csv

test_df = pd.read_csv("/Users/mvr/Documents/DSP Project SLD/sign_mnist_test.csv")
train_df = pd.read_csv("/Users/mvr/Documents/DSP Project SLD/sign_mnist_train.csv")
trainLabel=train_df['label']
testLabel=test_df['label']
del train_df['label']
del test_df['label']
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

model = Sequential()
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
model.add(Flatten())
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(units = 24 , activation = 'softmax'))
model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
model.summary()



# define the checkpoint
filepath = "/Users/mvr/Documents/DSP Project SLD/model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
#callbacks_list = [checkpoint]

model.fit(datagen.flow(x_train,TrainLabelEnc, batch_size = 128) ,epochs = 5 , validation_data = (x_test, TestLabelEnc),callbacks=checkpoint)


print("Accuracy of the model is - " , model.evaluate(x_test,TestLabelEnc)[1]*100 , "%")
