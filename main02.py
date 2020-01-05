#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author：Kung
# time:1/1/20 3:53 PM

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

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# print(os.listdir("../input"))


#%%
import tensorflow as tf

# Set the seed for hash based operations in python
os.environ['PYTHONHASHSEED'] = '0'

# Set the numpy seed
np.random.seed(111)

# Disable multi-threading in tensorflow ops
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

# Set the random seed in tensorflow at graph level
tf.set_random_seed(111)

# Define a tensorflow session with above session configs
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)

# Set the session in keras
K.set_session(sess)

# Make the augmentation sequence deterministic
aug.seed(111)


#%%
data_dir = Path('./data/chest_xray')

# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'

# Path to validation directory
val_dir = data_dir / 'val'

# Path to test directory
test_dir = data_dir / 'test'

#%%
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'

# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases_bactreia = pneumonia_cases_dir.glob('*bacteria*')
pneumonia_cases_virus = pneumonia_cases_dir.glob('*virus*')

# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []

# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    train_data.append((img,0))

# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases_bactreia:
    train_data.append((img, 1))

for img in pneumonia_cases_virus:
    train_data.append((img, 2))


# Get a pandas dataframe from the data we have in our list
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)

# Shuffle the data
train_data = train_data.sample(frac=1.).reset_index(drop=True)

# How the dataframe looks like?
train_data.head()

#%%
cases_count = train_data['label'].value_counts()
print(cases_count)

# Plot the results
plt.figure(figsize=(10,8))
sns.barplot(x=cases_count.index, y= cases_count.values)
plt.title('Number of cases', fontsize=14)
plt.xlabel('Case type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(range(len(cases_count.index)), ['Normal(0)', 'Bactreia(1)','Virus(2)'])
plt.show()

#%%
pneumonia_samples_v = (train_data[train_data['label']==2]['image'].iloc[:4]).tolist()
pneumonia_samples_b = (train_data[train_data['label']==1]['image'].iloc[:4]).tolist()
normal_samples = (train_data[train_data['label']==0]['image'].iloc[:4]).tolist()

# Concat the data in a single list and del the above two list
samples = pneumonia_samples_v + normal_samples + pneumonia_samples_b
del pneumonia_samples_v, normal_samples, pneumonia_samples_b

# Plot the data
f, ax = plt.subplots(3,4, figsize=(30,10))
for i in range(12):
    img = imread(samples[i])
    ax[i//4, i%4].imshow(img, cmap='gray')
    if i < 4:
        ax[i//4, i%4].set_title("Pneumonia_virus")
    elif i>=4 and i<8:
        ax[i // 4, i % 4].set_title("Normal")
    else:
        ax[i//4, i%4].set_title("Pneumonia_bacteria")
    ax[i//4, i%4].axis('off')
    ax[i//4, i%4].set_aspect('auto')
plt.show()


#%%
seq = iaa.OneOf([
    iaa.Fliplr(), # horizontal flips
    iaa.Affine(rotate=20), # roatation
    iaa.Multiply((1.2, 1.5))]) #random brightness

#%%
def data_gen(data, batch_size):
    # Get total number of samples in the data
    n = len(data)
    steps = n // batch_size

    # Define two numpy arrays for containing batch data and labels
    batch_data = np.zeros((batch_size, 224, 224, 3), dtype=np.float32)
    batch_labels = np.zeros((batch_size, 3), dtype=np.float32)

    # Get a numpy array of all the indices of the input data
    indices = np.arange(n)

    # Initialize a counter
    i = 0
    while True:
        np.random.shuffle(indices)
        # Get the next batch
        count = 0
        next_batch = indices[(i * batch_size):(i + 1) * batch_size]
        for j, idx in enumerate(next_batch):
            img_name = data.iloc[idx]['image']
            label = data.iloc[idx]['label']

            # one hot encoding
            encoded_label = to_categorical(label, num_classes=3)
            # read the image and resize
            img = cv2.imread(str(img_name))
            img = cv2.resize(img, (224, 224))

            # check if it's grayscale
            if img.shape[2] == 1:
                img = np.dstack([img, img, img])

            # cv2 reads in BGR mode by default
            orig_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # normalize the image pixels
            orig_img = img.astype(np.float32) / 255.

            batch_data[count] = orig_img
            batch_labels[count] = encoded_label

            # generating more samples of the undersampled class
            if label == 0 and count < batch_size - 2:
                aug_img1 = seq.augment_image(img)
                aug_img2 = seq.augment_image(img)
                aug_img1 = cv2.cvtColor(aug_img1, cv2.COLOR_BGR2RGB)
                aug_img2 = cv2.cvtColor(aug_img2, cv2.COLOR_BGR2RGB)
                aug_img1 = aug_img1.astype(np.float32) / 255.
                aug_img2 = aug_img2.astype(np.float32) / 255.

                batch_data[count + 1] = aug_img1
                batch_labels[count + 1] = encoded_label
                batch_data[count + 2] = aug_img2
                batch_labels[count + 2] = encoded_label
                count += 2

            else:
                count += 1

            if count == batch_size - 1:
                break

        i += 1
        yield batch_data, batch_labels

        if i >= steps:
            i = 0
#%%
def build_model():
    input_img = Input(shape=(224, 224, 3), name='ImageInput')
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_1')(input_img)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='Conv1_2')(x)
    x = MaxPooling2D((2, 2), name='pool1')(x)

    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_1')(x)
    x = SeparableConv2D(128, (3, 3), activation='relu', padding='same', name='Conv2_2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)

    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_1')(x)
    x = BatchNormalization(name='bn1')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_2')(x)
    x = BatchNormalization(name='bn2')(x)
    x = SeparableConv2D(256, (3, 3), activation='relu', padding='same', name='Conv3_3')(x)
    x = MaxPooling2D((2, 2), name='pool3')(x)

    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_1')(x)
    x = BatchNormalization(name='bn3')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_2')(x)
    x = BatchNormalization(name='bn4')(x)
    x = SeparableConv2D(512, (3, 3), activation='relu', padding='same', name='Conv4_3')(x)
    x = MaxPooling2D((2, 2), name='pool4')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dropout(0.7, name='dropout1')(x)
    x = Dense(512, activation='relu', name='fc2')(x)
    x = Dropout(0.5, name='dropout2')(x)
    x = Dense(3, activation='softmax', name='fc3')(x)

    model = Model(inputs=input_img, outputs=x)
    return model

#%%
model =  build_model()
model.summary()

#%%
from keras.models import load_model
model=load_model('my_model3.h5') #model3 is 3 classes

#%%
opt = Adam(lr=0.0001, decay=1e-5)
es = EarlyStopping(patience=5)
chkpt = ModelCheckpoint(filepath='best_model_todate', save_best_only=True, save_weights_only=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer=opt)

#%%
batch_size=64
nb_epochs = 50
# Get a train data generator
train_data_gen = data_gen(data=train_data, batch_size=batch_size)

# Define the number of training steps
nb_train_steps = train_data.shape[0]//batch_size

print("Number of training and validation steps: {} and {}".format(nb_train_steps, len(valid_data)))

#%%
# Fit the model
training_history = model.fit_generator(train_data_gen, epochs=nb_epochs, steps_per_epoch=nb_train_steps,
                              callbacks=[es, chkpt],
                              )

#%%
fig, ax1 = plt.subplots(figsize=(9,5))
plt.title('Deepwise model')
plt.xlabel('Epochs')
ax2 = ax1.twinx()
ax1.set_ylabel('Loss', color='tab:red')
ax2.set_ylabel('Accuracy', color='tab:blue')
curve1, = ax1.plot(training_history.history['loss'], label="Loss", color='r')
curve2, = ax2.plot(training_history.history['acc'], label="Accuracy", color='b')
curves=[curve1,curve2]

ax1.legend(curves, [curve.get_label() for curve in curves],loc='right')

plt.show()
plt.savefig('loss3.png')

model.save('my_model3.h5')

#%%
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
# Evaluation on test dataset
test_loss, test_score = model.evaluate(test_data, test_labels, batch_size=16)
print("Loss on test set: ", test_loss)
print("Accuracy on test set: ", test_score)

#%%
preds = model.predict(test_data, batch_size=16)
preds = np.argmax(preds, axis=-1)

# Original labels
orig_test_labels = np.argmax(test_labels, axis=-1)

print(orig_test_labels.shape)
print(preds.shape)

#%%
cm  = confusion_matrix(orig_test_labels, preds)
# plt.figure()
plot_confusion_matrix(cm,figsize=(6,4), hide_ticks=True,cmap=plt.cm.Blues)
plt.xticks(range(3), ['Normal(0)', 'Bactreia(1)','Virus(2)'], fontsize=16)
plt.yticks(range(3), ['Normal(0)', 'Bactreia(1)','Virus(2)'], fontsize=16)
plt.show()

tn, fp, fn, tp = cm.ravel()
precision = tp/(tp+fp)
recall = tp/(tp+fn)
print("Recall of the model is {:.2f}".format(recall))
print("Precision of the model is {:.2f}".format(precision))


#%%
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sb
figure = sb.heatmap(cm, annot=True, annot_kws={"size": 16}, fmt='d', cmap='YlGnBu')
figure.set_xticklabels(['Normal', 'Bacteria','Virus'])
figure.set_yticklabels(['Normal', 'Bacteria','Virus'])
# score = round(rfc.score(Xtest, Ytest), 6)
# plt.title('Testing Total Accuracy = ' + str(score))
# plt.yticks(rotation=0)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

#%%
from keras.utils import np_utils
from sklearn import metrics
from sklearn.metrics import roc_curve, auc

y_pred_proba = model.predict(test_data, batch_size=16)
orig_test_labels=np_utils.to_categorical(orig_test_labels)
#%%

#N-roc
C=['Normal', 'Bacteria','Virus']
for i in range(3):
    fpr, tpr, _ = metrics.roc_curve(orig_test_labels[:,i],y_pred_proba[:, i])
    auc = metrics.roc_auc_score(orig_test_labels[:,i],y_pred_proba[:, i])
    plt.plot(fpr,tpr,label=C[i]+':' +str(np.round(auc,4)))


    # plt.show()
plt.legend(loc=4)
plt.title('ROC')
plt.grid()



