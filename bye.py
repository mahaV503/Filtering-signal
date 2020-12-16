
#This is the DSP lab project done by Anshi,Megha and me
import time
import uuid
import cv2
import os
import numpy as np

import PIL
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
test_df = pd.read_csv("/Users/mvr/Documents/DSP Project SLD/sign_mnist_test.csv")
testLabel=test_df['label']
del test_df['label']
#NOw encoding the labels 
labelEncoding = LabelBinarizer()
TestLabelEnc = labelEncoding.fit_transform(testLabel)

imgsTest = test_df.values
x_test = imgsTest / 255
x_test = x_test.reshape(-1,28,28,1)
filepath = "/Users/mvr/Documents/DSP Project SLD/model.h5"

new_model = load_model(filepath)

#print("Accuracy of the model is - " , new_model.evaluate(x_test,TestLabelEnc)[1]*100 , "%")

cap=cv2.VideoCapture(0)
box_size=234
width=int(cap.get(3))

while True:
    ret,frame=cap.read()
    frame=cv2.flip(frame,1)
    cv2.rectangle(frame, (width - box_size, 0), (width, box_size), (0, 250, 150), 2)
    cv2.namedWindow("Sign Language Detection", cv2.WINDOW_NORMAL)
    roi = frame[5: box_size-5 , width-box_size + 5: width -5]
    #imagesmall=load_img(roi, grayscale=True, target_size=(28, 28))
    #img = img_to_array(imagesmall)
    #img=img.reshape(1,28,28,1)
    #img=img.astype('float32')
    #img=img/255
    #imageTen=tf.convert_to_tensor(np.expand_dims(imageNP,0),dtype=tf.float32)
    #roi = np.array([roi]).astype('float64') / 255.0
    #roii=np.reshape(roi,(1,784))
    #imageT=tf.reshape(1,784)
    print(roi.shape)
    img = np.mean(roi, axis=2).reshape(-1,28,28,1)
    print(img.shape)
    #image = tf.keras.preprocessing.image.load_img(roi, target_size=(28, 28),color_mode='grayscale')
    #input_arr = keras.preprocessing.image.img_to_array(image)
    #input_arr = np.array([img])  # Convert single image to a batch.
    predictions = new_model.predict(img)
    print("Debug")
    #pred=new_model.predict(img)


    target_index = np.argmax(predictions[0])
    cv2.putText(frame, "prediction: {} {:.2f}%".format(label_names[np.argmax(pred[0])], prob*100 ),(10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Sign Language Detection", frame)

    k = cv2.waitKey(1)

    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
