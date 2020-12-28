# -*- coding: utf-8 -*-
"""
Tensorflow #1
@author: Deep.I Inc. @Jongwon Kim
Revision date: 2020-12-13
See here for more information :
    https://deep-eye.tistory.com
    https://deep-i.net
"""

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.utils import plot_model

# other models
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.resnet import ResNet50
# from tensorflow.keras.applications.vgg19 import VGG19

# Loading Pre-trained weight 
model = VGG16(weights='imagenet')

# Weight extraction
weight_1 , bias_1 = model.layers[1].get_weights()

# Summary
print(model.summary())
# Flowchart
plot_model(model, to_file='test.png')