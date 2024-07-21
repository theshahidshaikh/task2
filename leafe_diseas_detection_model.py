import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sna

# DATA PROCESSING

# Training Image Preprocessing

print('preprocessing training')
training_dataset = tf.keras.utils.image_dataset_from_directory(
    'E:\Bharti Vidhypith\INTERNSHIP\Deep Learning projects\Plant Diseases Dataset\\train',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# Validation Image Preprocessing
print('preprocessing valid')
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    'E:\Bharti Vidhypith\INTERNSHIP\Deep Learning projects\Plant Diseases Dataset\\valid',
    labels="inferred",
    label_mode="categorical",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
)

# MODEL BUILDING

from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten,Dropout
from tensorflow.keras.models import Sequential

model = Sequential()

# Building Convolution Layer

model.add(Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', input_shape=[128,128,3]))
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=128, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=256, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Conv2D(filters=512, kernel_size=3, padding='same', activation='relu'))
model.add(Conv2D(filters=512, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=2,strides=2))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=1240,activation='relu'))
model.add(Dropout(0.4))
# Output Layer
model.add(Dense(units=11,activation='softmax'))

# COMPILING MODEL
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# MODEL TRAINING
training_history = model.fit(x=training_dataset,validation_data=validation_dataset,epochs=5)

#SAVING MODEL
model.save("Train_model.keras")



import json

with open("training_hist.json","w") as f:
    json.dump(training_history.history,f)


