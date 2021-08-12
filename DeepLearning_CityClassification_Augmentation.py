# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 10:02:48 2021

@author: m252047
"""

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import time

#%% Data loading / augmentation 

datagen = ImageDataGenerator(
        rotation_range = 45,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'reflect')

chicago = io.imread(r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\chicago\downtown_chicago.jpg')

miami = io.imread(r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\miami\downtown_miami.jpg')

newyork = io.imread(r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\newyork\downtown_newyork.jpg')

chicago = chicago.reshape((1, ) + chicago.shape) # 1, x, y, 3

miami = miami.reshape((1, ) + miami.shape)

newyork = newyork.reshape((1, ) + newyork.shape)

    
def read_images():
    import os
    import numpy as np
    from PIL import Image
    dataset = []
    size = 128
    img_dir = r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation2' #Change This
    my_imgs = os.listdir(img_dir)
    # print(my_imgs)
    global img
    global x
    for index, img_name in enumerate(my_imgs[1:]):
        if (img_name.split('.')[1] == 'jpg'):
            img = io.imread(img_dir + '\\' + img_name)
            img = Image.fromarray(img, 'RGB')
            img = img.resize((size, size))
            dataset.append(np.array(img))
    x = np.array(dataset)
    
def aug_gen( s): # aug_gen function, select is going to be either chicago miami or newyork
    if s == 'chicago' or s == 'Chicago':
        i = 0
        print('Augmenting {}...'.format(s))
        time.sleep(2)
        for batch in datagen.flow(chicago, batch_size = 4,
                          save_to_dir=r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\Augmented', #Change this
                          save_prefix = 'aug',
                          save_format = 'png'):
                i += 1
                if i > 5:
                    break
    elif s == 'Miami' or s == 'miami':
        i = 0
        print('Augmenting {}...'.format(s))
        time.sleep(2)
        for batch in datagen.flow(miami, batch_size = 4,
                          save_to_dir=r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\Augmented', #Change this
                          save_prefix = 'aug',
                          save_format = 'png'):
                i += 1
                if i > 5:
                    break
    elif s == 'new york' or s == 'New York' or s == 'newyork' or s == 'NewYork' or s == 'Newyork':
         i = 0
         print('Augmenting {}...'.format(s))
         time.sleep(2)
         for batch in datagen.flow(newyork, batch_size = 4,
                          save_to_dir=r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\Augmented',
                          save_prefix = 'aug',
                          save_format = 'png'):
                i += 1
                if i > 5:
                    break
    else:
         i = 0
         print('Augmenting {}...'.format(s))
         time.sleep(2)
         for batch in datagen.flow(s, batch_size = 7,
                          save_to_dir=r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation2\Augmented',
                          save_prefix = 'aug',
                          save_format = 'png'):
                i += 1
                if i > 5:
                    break   
    
def flow_from_dir( dir, save_dir):
    i = 0
    print('Augmenting...')
    time.sleep(2)
    for batch in datagen.flow_from_directory(directory = dir,
                                             batch_size = 16,
                                             target_size = (256, 256),
                                             color_mode = 'rgb',
                                             save_to_dir = save_dir,
                                             save_prefix = 'AUG',
                                             save_format = 'png'):
        i += 1
        if i > 31:
            break
        
        
#%% Classes:
# 0. Chicago
# 1. Miami
# 2. New York

import sklearn 
from sklearn.model_selection import train_test_split, ShuffleSplit
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

data_dir = r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation'
categories = ['Chicago', 'Miami', 'NewYork']

# path_imgs = r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\Augmented'

training_data = []
for category in categories:
    path = os.path.join(data_dir, category)
    class_num = categories.index(category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            plt.imshow(img_array, cmap='gray')
            training_data.append([img_array, class_num])
        except Exception as e:
            pass
 
import random 
random.shuffle(training_data)

for sample in training_data:
    print(sample[1])
x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
    
    
x = np.array(x).reshape(-1, 128, 128, 1)
    
import pickle
pickle_out = open('x.pickle', 'wb')
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open('y.pickle', 'wb')
pickle.dump(y, pickle_out)
pickle_out.close()

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import pickle

x = pickle.load(open('x.pickle', 'wb'))
y = pickle.load(open('y.pickle', 'wb'))

x = x/255.0

model = Sequential()
activation = 'sigmoid'
model.add(Conv2D(32, 3, activation = activation, padding = 'same', input_shape = x.shape[1:]))
model.add(BatchNormalization())


model.add(Conv2D(32, 3, activation = activation, padding = 'same', kernel_initializer= 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer= 'he_uniform'))
model.add(BatchNormalization())

model.add(Conv2D(64, 3, activation = activation, padding = 'same', kernel_initializer= 'he_uniform'))
model.add(BatchNormalization())
model.add(MaxPooling2D())


model.add(Flatten())
model.add(Dense(128, activation = activation, padding = 'same', kernel_intializer = 'he_uniform'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])
print(model.summary())



#%%

# if __name__ == '__main__':
#     # input2 = input('Enter a city: ')
#     # read_images()
#     # aug_gen( x)
    
#     dir = r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation'
#     save_dir = r'\\mfad\researchmn\ULTRASOUND\FATEMI\MEMBERS\PUTHAWALA\Augmentation\Augmented'

#     # flow_from_dir( dir, save_dir)
#     # create_training_data()
#     # print(len(training_data))
    
#     create_model()
    

    
    