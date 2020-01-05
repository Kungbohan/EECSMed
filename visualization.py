#!/usr/bin/env python 
# -*- coding: utf-8 -*- 
# author：Kung 
# time:1/5/20 7:30 PM

import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize
from keras.models import Sequential, Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, SeparableConv2D
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import cv2
from keras import backend as K
color = sns.color_palette()

#%%
data_dir = Path('./data/chest_xray')
test_dir = data_dir / 'test'

# Preparing test data
normal_cases_dir = test_dir / 'NORMAL'
pneumonia_cases_dir = test_dir / 'PNEUMONIA'

# normal_cases = normal_cases_dir.glob('*.jpeg')
# pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')

normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases_bactreia = pneumonia_cases_dir.glob('*bacteria*')
pneumonia_cases_virus = pneumonia_cases_dir.glob('*virus*')

test_data = []
test_labels = []

for img in normal_cases:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(0, num_classes=3)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases_bactreia:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(1, num_classes=3)
    test_data.append(img)
    test_labels.append(label)

for img in pneumonia_cases_virus:
    img = cv2.imread(str(img))
    img = cv2.resize(img, (224, 224))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.
    label = to_categorical(2, num_classes=3)
    test_data.append(img)
    test_labels.append(label)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

print("Total number of test examples: ", test_data.shape)
print("Total number of labels:", test_labels.shape)

#%%
import os
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np

from keras.models import load_model
model=load_model('my_model3.h5') # model1 is normal deepwise

#%%
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)
true_label = np.argmax(test_labels,axis=-1)
ddic={0:'Normal',1:"Pneumonia_bacteria",2:"Pneumonia_virus"}

#%%
dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Conv4_3').output) #Conv4_3 Conv1_2

dense1_output = dense1_layer_model.predict(test_data[5:6, :, :, :])
f, ax = plt.subplots(3,4, figsize=(30,10),)
f.subplots_adjust(wspace =0.1, hspace =0.1)
for i in range(11):
    ax[i // 4, i % 4].imshow(dense1_output[0,:,:,i], cmap='gray')
    ax[i // 4, i % 4].axis('off')
ax[2,3].imshow(np.squeeze(test_data[5:6, :, :, :]), cmap='gray')
ax[2,3].axis('off')


#%%
fc_layer_model = Model(inputs=model.input,outputs=model.get_layer('fc3').output) # fc1 fc2
fc_output = fc_layer_model.predict(test_data)
from sklearn.manifold import TSNE
X_tsne = TSNE(n_components=3).fit_transform(fc_output)

#%%
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)
plt.figure(figsize=(8, 8))
for i in range(X_norm.shape[0]):
    plt.text(X_norm[i, 0], X_norm[i, 1], str(true_label[i]), color=plt.cm.Set1(true_label[i]),
             fontdict={'weight': 'bold', 'size': 9})
plt.xticks([])
plt.yticks([])
plt.show()
