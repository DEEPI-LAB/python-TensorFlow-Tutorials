# -*- coding: utf-8 -*-
"""
Tensorflow #3
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-13
See here for more information :
    https://deep-eye.tistory.com
    https://deep-i.net
"""

import cv2
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.applications.vgg16 import preprocess_input, decode_predictions

# loading weights from pre-trained model 
model = VGG16(weights='imagenet')

# loading images for classification
image = cv2.imread("sample/cat.jpg")
image = cv2.resize(image,dsize=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
image = preprocess_input(image)

# class prediction
yhat = model.predict(image)
label = decode_predictions(yhat)

label = label[0][0]

# result
print("%s (%.2f%%)" % (label[1], label[2]*100))
# display image
image = cv2.imread("sample/cat.jpg")
cv2.imshow(label[1],image)
cv2.waitKey()
