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

# 사전 학습된 모델 불러오기
model = VGG16(weights='imagenet')

# 분류를 위한 이미지 불러오기
image = cv2.imread("sample/cat.jpg")
image = cv2.resize(image,dsize=(224,224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0],image.shape[1],image.shape[2]))
image = preprocess_input(image)

# 클래스 예측
yhat = model.predict(image)
# 클래스 디코딩 (STRING)
label = decode_predictions(yhat)
# 클래스 부여 (최상위 스코어 클래스)
label = label[0][0]
# Result
print("%s (%.2f%%)" % (label[1], label[2]*100))
# Image Show
image = cv2.imread("sample/cat.jpg")
cv2.imshow(label[1],image)
cv2.waitKey()
