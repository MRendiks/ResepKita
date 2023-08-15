# ResepKita
> ResepKita Native Application built using the Kotlin programming language and python using Flask. This application was created with the aim of making it easier for search a food recipe and its ingredients easily and buy all food ingredients just from scanned food image.


## Usage
### 1. Create Virtual Environtment
```
python -m venv name_of_env
```
### 2. install depedency for project
```
pip install -r requirements.txt
```
### 3. run  project
```
python app.py
```

## Problem

What are the problems that made us make this application?
 - People wanted to do something without wasting too much time, especially cooking. The problem is that buying ingredients for cooking takes too much time. They have to think about what they need to cook, go to the supermarket, and then spend time walking around to find the ingredients they need.
 - People doesn't know what the name of the food is, the recipe for a food is just from the appearance of the food.

## Solution

**ResepKita** is here to solve this problem. With this application, wanted to cook by themselves. But because of their time-consuming and busy activities, they never started to shop for cooking in the first place. They also have a hard time choosing what to cook. We want to solve this problem by creating an application where we provide recipes and you can automatically buy its ingredients online.

## Apps Features

App features :
1. Register user.
2. Login user.
3. Profile info.
4. History info.
5. Cart menu.

## Overview


## Google Colab
> [ResepKita Notebook](https://colab.research.google.com/drive/1jEug6xCwtNVDyaqfLSJamrPCkobd3K3_?usp=sharing)

Importing Module
---

``` {.python}
import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.models import Sequential
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation,GlobalMaxPooling2D
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
```

Connecting Google Drive
---
``` {.python}
# Menyambungkan ke GDrive
from google.colab import drive
drive.mount('/content/gdrive')
```

Mounted at /content/gdrive

Extracting Dataset
---
``` {.python}
local_zip = '/content/gdrive/MyDrive/datasets_4.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/dataset')
zip_ref.close()
```

``` {.python}
os.chdir('/content/dataset')
base_url = '/content/dataset/datasets_4/content/datasets'
```

Creating Generator
---
``` {.python}
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255., 
                             rotation_range = 40, 
                             horizontal_flip = True, 
                             validation_split=0.3,
                             fill_mode = 'nearest')

# train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 40, horizontal_flip = True, validation_split=0.3)
# validation_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = datagen.flow_from_directory(
        base_url,
        classes = ['Chocolate Cake', 'Pasta', 'Pizza', 'Salad'],
        target_size=(128, 128),
        subset='training',
        batch_size=32,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(
        base_url,
        classes = ['Chocolate Cake', 'Pasta', 'Pizza', 'Salad'],
        target_size=(128, 128),
        subset='validation',
        batch_size=32,
        class_mode='categorical',
        shuffle=False)
```

Found 559 images belonging to 4 classes.<br>
Found 239 images belonging to 4 classes.

Creating Model
---
``` {.python}
image_size = 128
input_shape = (image_size, image_size, 3)

epochs = 20
batch_size = 32

base_model = VGG16(input_shape=input_shape, include_top=False, weights="imagenet")
    
for layer in base_model.layers[:15]:
    layer.trainable = False

for layer in base_model.layers[15:]:
    layer.trainable = True
    
last_layer = base_model.get_layer('block5_pool')
last_output = last_layer.output
    
# Flatten the output layer to 1 dimension
x = GlobalMaxPooling2D()(last_output)
# Add a fully connected layer with 512 hidden units and ReLU activation
x = Dense(512, activation='relu')(x)
# Add a dropout rate of 0.5
x = Dropout(0.5)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(4, activation='softmax')(x)

model = Model(base_model.input, x)

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])

model.summary()
```

Downloading data from <br>
https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5<br>
58892288/58889256 [==============================] - 0s 0us/step<br>
58900480/58889256 [==============================] - 0s 0us/step<br>
<br>
    _________________________________________________________________<br>
     Layer (type)                Output Shape              Param #   <br>
    _________________________________________________________________<br>
     input_1 (InputLayer)        [(None, 128, 128, 3)]     0         <br>
                                                                     <br>
     block1_conv1 (Conv2D)       (None, 128, 128, 64)      1792      <br>
                                                                     <br>
     block1_conv2 (Conv2D)       (None, 128, 128, 64)      36928     <br>
                                                                     <br>
     block1_pool (MaxPooling2D)  (None, 64, 64, 64)        0         <br>
                                                                     <br>
     block2_conv1 (Conv2D)       (None, 64, 64, 128)       73856     <br>
                                                                     <br>
     block2_conv2 (Conv2D)       (None, 64, 64, 128)       147584    <br>
                                                                     <br>
     block2_pool (MaxPooling2D)  (None, 32, 32, 128)       0         <br>
                                                                     <br>
     block3_conv1 (Conv2D)       (None, 32, 32, 256)       295168    <br>
                                                                     <br>
     block3_conv2 (Conv2D)       (None, 32, 32, 256)       590080    <br>
                                                                     <br>
     block3_conv3 (Conv2D)       (None, 32, 32, 256)       590080    <br>
                                                                     <br>
     block3_pool (MaxPooling2D)  (None, 16, 16, 256)       0         <br>
                                                                     <br>
     block4_conv1 (Conv2D)       (None, 16, 16, 512)       1180160   <br>
                                                                     <br>
     block4_conv2 (Conv2D)       (None, 16, 16, 512)       2359808   <br>
                                                                     <br>
     block4_conv3 (Conv2D)       (None, 16, 16, 512)       2359808   <br>
                                                                     <br>
     block4_pool (MaxPooling2D)  (None, 8, 8, 512)         0         <br>
                                                                     <br>
     block5_conv1 (Conv2D)       (None, 8, 8, 512)         2359808   <br>
                                                                     <br>
     block5_conv2 (Conv2D)       (None, 8, 8, 512)         2359808   <br>
                                                                     <br>
     block5_conv3 (Conv2D)       (None, 8, 8, 512)         2359808   <br>
                                                                     <br>
     block5_pool (MaxPooling2D)  (None, 4, 4, 512)         0         <br>
                                                                     <br>
     global_max_pooling2d (Globa  (None, 512)              0         <br>
     lMaxPooling2D)                                                  <br>
                                                                     <br>
     dense (Dense)               (None, 512)               262656    <br>
                                                                     <br>
     dropout (Dropout)           (None, 512)               0         <br>
                                                                     <br>
     dense_1 (Dense)             (None, 4)                 2052      <br>
                                                                     <br>
    _________________________________________________________________<br>
    Total params: 14,979,396                                         <br>
    Trainable params: 7,344,132                                      <br>
    Non-trainable params: 7,635,264                                  <br>
    _________________________________________________________________<br>




``` {.python}
history = model.fit(train_generator, 
                    validation_data = validation_generator, 
                    epochs = epochs)
```

{.output .stream .stdout}<br>
    Epoch 1/20<br>
    18/18 [==============================] - 147s 8s/step - loss: 1.2864 - accuracy: 0.4240 - val_loss: 0.7796 - val_accuracy: 0.7322<br>
    Epoch 2/20<br>
    18/18 [==============================] - 144s 8s/step - loss: 0.8153 - accuracy: 0.6583 - val_loss: 0.4149 - val_accuracy: 0.8577<br>
    Epoch 3/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.4909 - accuracy: 0.8265 - val_loss: 0.2798 - val_accuracy: 0.9079<br>
    Epoch 4/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.3548 - accuracy: 0.8658 - val_loss: 0.2479 - val_accuracy: 0.8996<br>
    Epoch 5/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.2827 - accuracy: 0.9034 - val_loss: 0.2534 - val_accuracy: 0.8828<br>
    Epoch 6/20<br>
    18/18 [==============================] - 142s 8s/step - loss: 0.2910 - accuracy: 0.8891 - val_loss: 0.2121 - val_accuracy: 0.9163<br>
    Epoch 7/20<br>
    18/18 [==============================] - 142s 8s/step - loss: 0.2292 - accuracy: 0.9338 - val_loss: 0.2024 - val_accuracy: 0.9121<br>
    Epoch 8/20<br>
    18/18 [==============================] - 142s 8s/step - loss: 0.1673 - accuracy: 0.9410 - val_loss: 0.1878 - val_accuracy: 0.9247<br>
    Epoch 9/20<br>
    18/18 [==============================] - 142s 8s/step - loss: 0.1363 - accuracy: 0.9535 - val_loss: 0.1817 - val_accuracy: 0.9247<br>
    Epoch 10/20<br>
    18/18 [==============================] - 141s 8s/step - loss: 0.1178 - accuracy: 0.9553 - val_loss: 0.2543 - val_accuracy: 0.8912<br>
    Epoch 11/20<br>
    18/18 [==============================] - 141s 8s/step - loss: 0.0799 - accuracy: 0.9642 - val_loss: 0.2377 - val_accuracy: 0.9163<br>
    Epoch 12/20<br>
    18/18 [==============================] - 141s 8s/step - loss: 0.0867 - accuracy: 0.9678 - val_loss: 0.1820 - val_accuracy: 0.9372<br>
    Epoch 13/20<br>
    18/18 [==============================] - 141s 8s/step - loss: 0.0586 - accuracy: 0.9911 - val_loss: 0.2226 - val_accuracy: 0.9289<br>
    Epoch 14/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.0656 - accuracy: 0.9714 - val_loss: 0.2066 - val_accuracy: 0.9038<br>
    Epoch 15/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.0413 - accuracy: 0.9875 - val_loss: 0.1945 - val_accuracy: 0.9331<br>
    Epoch 16/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.0208 - accuracy: 0.9982 - val_loss: 0.1887 - val_accuracy: 0.9247<br>
    Epoch 17/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.0461 - accuracy: 0.9821 - val_loss: 0.2135 - val_accuracy: 0.9289<br>
    Epoch 18/20<br>
    18/18 [==============================] - 143s 8s/step - loss: 0.0344 - accuracy: 0.9893 - val_loss: 0.1563 - val_accuracy: 0.9372<br>
    Epoch 19/20<br>
    18/18 [==============================] - 144s 8s/step - loss: 0.0484 - accuracy: 0.9821 - val_loss: 0.3451 - val_accuracy: 0.9079<br>
    Epoch 20/20<br>
    18/18 [==============================] - 141s 8s/step - loss: 0.0661 - accuracy: 0.9732 - val_loss: 0.3305 - val_accuracy: 0.8954<br>

``` {.python}
model.evaluate(validation_generator)
```

{.output .stream .stdout}<br>
8/8 [==============================] - 37s 4s/step - loss: 0.2959 - accuracy: 0.9163<br>
[0.2958613336086273, 0.9163179993629456]<br>

``` {.python}
STEP_SIZE_TEST=validation_generator.n//validation_generator.batch_size
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)
```

{.output .stream .stdout}<br>
8/8 [==============================] - 44s 5s/step<br>

Define a Function to Plot
---

``` {.python}
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

## Dataset
> [ResepKita Dataset](https://drive.google.com/file/d/1PPhcIdtS0W69bR6XRrLrKebUrZlWnFT8/view?usp=sharing)
