# -*- coding: utf-8 -*-
"""
Tensorflow #2
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-27
See here for more information :
    https://deep-eye.tistory.com
    https://deep-i.net
"""

import cv2
from matplotlib import pyplot
from tensorflow.keras.models import Model
from tensorflow.python.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.python.keras.preprocessing.image import img_to_array

# loading weights from pre-trained model 
base_model = VGG16(weights='imagenet')
base_model.summary()
# model separation (CNN / FC)
model = Model(inputs = base_model.input,outputs = base_model.get_layer('block5_conv3').output)
model.summary()

# loading image
image = cv2.imread("1.jpg")
image = cv2.resize(image,dsize=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
image = preprocess_input(image)
# feature extraction
feature_map = model.predict(image)

# plot size (square x square)

square = 8
ix = 1
for i in range(square):
    for j in range(square):
        ax = pyplot.subplot(square,square,ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # feature Map
        pyplot.imshow(feature_map[0,:,:,ix-1])
        ix = ix + 1
pyplot.show()
