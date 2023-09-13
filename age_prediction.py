import pandas as pd
import tensorflow as tf
import cv2
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

width=height=48
batch_size=4
EPOCH=20

# images=[]
# ages=[]
#
# for image_name in os.listdir(r'C:\Users\Ariya Rayaneh\Desktop\archive (36)\UTKFace'):
#     age=int(image_name.split('_')[0])
#     ages.append(int(image_name.split('_')[0]))
#
#     image=cv2.imread(rf'C:\Users\Ariya Rayaneh\Desktop\archive (36)\UTKFace\{image_name}')
#
#     image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#     image=cv2.resize(image,(width,height))
#     images.append(image)
#
# images=pd.Series(images,name='images')
# ages=pd.Series(ages,name='ages')
#
# dataframe=pd.concat([images,ages],axis=1)
#
# under_four=dataframe[dataframe['ages']>=4]
# under_four=dataframe[dataframe['ages']<80]
#
# X=np.array(dataframe['images'].values.tolist())
# Y=np.array(dataframe['ages'].values.tolist())
#
# print(X.shape)
# print(Y.shape)
#
# X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.2)
#
# print(X_train.shape)
# print(X_val.shape)
# print(Y_train.shape)
# print(Y_val.shape)
#
# idg=ImageDataGenerator(rescale=1/255,
#                        horizontal_flip=True
#                        )
#
# train_data=idg.flow(X_train,Y_train,batch_size=batch_size)
# val_data=idg.flow(X_val,Y_val,batch_size=batch_size)
#
# base_model=keras.applications.ResNet50V2(input_shape=(width,height,3),
#                                           weights='imagenet',
#                                           include_top=False,
#                                           pooling='avg'
#                                           )
#
# for layer in base_model.layers[:-4]:
#     layer.trainable=False
#
# model=keras.Sequential([
#     base_model,
#     Dropout(0.5),
#     Dense(1,activation='relu')
# ])
#
# model.compile(optimizer=Adam(learning_rate=0.001),loss=tf.keras.losses.mse)
#
#
# H = model.fit(
#     train_data,
# 	steps_per_epoch=len(train_data) // batch_size,
# 	validation_data=val_data,
# 	validation_steps=len(val_data) // batch_size,
# 	epochs=EPOCH)
#
#
# model.save(r'C:\Users\Ariya Rayaneh\Desktop\my_model_new.h5',save_format="h5")

new_model =keras.models.load_model(r'C:\Users\Ariya Rayaneh\Desktop\my_model_new.h5')
new_model.summary()

image=cv2.imread(r'C:\Users\Ariya Rayaneh\Desktop\face37.jpg')
img = img_to_array(image,data_format='channels_first')
image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
image=cv2.resize(image,(width,height))
image=image/255.0

image=image[np.newaxis,...]
result=new_model.predict(image)
print(result)
