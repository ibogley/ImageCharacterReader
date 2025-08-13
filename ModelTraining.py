import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential 

dataDir = '/Users/ianbogley/Desktop/Data Science/image recognition/Image Character Reader/data/2025-08-12'
os.chdir('/Users/ianbogley/Desktop/Data Science/image recognition/Image Character Reader')


train = tf.keras.utils.image_dataset_from_directory(
    dataDir,
    validation_split = .2,
    subset="training",
    seed=123,
    image_size=(100,100),
    batch_size=10
)

test = tf.keras.utils.image_dataset_from_directory(
    dataDir,
    validation_split = .75,
    subset="validation",
    seed=123,
    image_size=(100,100),
    batch_size=10
)

class_names=train.class_names


#See a set of pictures
plt.figure(figsize=(10,10))
for images, labels in train.take(1):
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype('uint8'))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()
AUTOTUNE = tf.data.AUTOTUNE

train = train.cache().shuffle(100).prefetch(buffer_size=AUTOTUNE)
test = test.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Model 0: 2 convolutional layers
model = Sequential([
    layers.Rescaling(1/255,input_shape=(100,100,3)),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(12,activation = 'relu'),
    layers.Dense(num_classes)
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

epochs = 10
history = model.fit(
    train,
    validation_data = test,
    epochs = epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions = model.predict(test)
prediction_classes = [class_names[np.argmax(pred)] for pred in predictions]
true_labels = [class_names[y] for x,y in test.unbatch()]

errors = [not pred==label for pred,label in zip(prediction_classes,true_labels)]
errors_index = [i for i in range(len(errors)) if errors[i]]

test_unbatched = list(test.unbatch())
len(test_unbatched)
plt.figure(figsize=(10,10))

for i in errors_index[0:5]:
    plt.imshow(list(test_unbatched)[i][0].numpy().astype(int))
    plt.title(class_names[list(test_unbatched)[i][1]])
    plt.xlabel('prediction: '+prediction_classes[i])
    plt.show()

#Accuracy: roughly 50%

#Model 1: 2 convolutional layers, increased stats
model1 = Sequential([
    layers.Rescaling(1/255,input_shape=(100,100,3)),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(186,activation = 'relu'),
    layers.Dense(num_classes)
])

model1.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

history1 = model1.fit(
    train,
    validation_data = test,
    epochs = epochs
)

acc1 = history1.history['accuracy']
val_acc1 = history1.history['val_accuracy']

loss1 = history1.history['loss']
val_loss1 = history1.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc1, label='Training Accuracy')
plt.plot(epochs_range, val_acc1, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss1, label='Training Loss')
plt.plot(epochs_range, val_loss1, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions1 = model1.predict(test)
prediction_classes1 = [class_names[np.argmax(pred)] for pred in predictions1]

errors1 = [not pred==label for pred,label in zip(prediction_classes1,true_labels)]
errors_index1 = [i for i in range(len(errors1)) if errors1[i]]

plt.figure(figsize=(10,10))
for i in errors_index1[0:5]:
    plt.imshow(list(test_unbatched)[i][0].numpy().astype(int))
    plt.title(class_names[list(test_unbatched)[i][1]])
    plt.xlabel('prediction: '+prediction_classes1[i])
    plt.show()
#Accuracy: roughly 80%

#Model 2: 3 convolutional layers, increased stats,additional convolutional layers
#Dont do too many max pooling, seems to cause issues
model2 = Sequential([
    layers.Rescaling(1/255,input_shape=(100,100,3)),
    layers.Conv2D(256,3,padding='same',activation='relu'),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(186,activation = 'relu'),
    layers.Dense(num_classes)
])

model2.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

history2 = model2.fit(
    train,
    validation_data = test,
    epochs = epochs
)

acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc2, label='Training Accuracy')
plt.plot(epochs_range, val_acc2, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss2, label='Training Loss')
plt.plot(epochs_range, val_loss2, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions2 = model2.predict(test)
prediction_classes2 = [class_names[np.argmax(pred)] for pred in predictions2]

errors2 = [not pred==label for pred,label in zip(prediction_classes2,true_labels)]
errors_index2 = [i for i in range(len(errors2)) if errors2[i]]

plt.figure(figsize=(10,10))
for i in errors_index2[0:5]:
    plt.imshow(list(test_unbatched)[i][0].numpy().astype(int))
    plt.title(class_names[list(test_unbatched)[i][1]])
    plt.xlabel('prediction: '+prediction_classes2[i])
    plt.show()

#Accuracy: way lower


#Model 3: 2 convolutional layers, increased stats, additional layer
model2 = Sequential([
    layers.Rescaling(1/255,input_shape=(100,100,3)),
    layers.MaxPooling2D(),
    layers.Conv2D(256,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128,3,padding='same',activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(186,activation = 'relu'),
    layers.Dense(num_classes)
])

model2.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics =['accuracy']
)

history2 = model2.fit(
    train,
    validation_data = test,
    epochs = epochs
)

acc2 = history2.history['accuracy']
val_acc2 = history2.history['val_accuracy']

loss2 = history2.history['loss']
val_loss2 = history2.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc2, label='Training Accuracy')
plt.plot(epochs_range, val_acc2, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss2, label='Training Loss')
plt.plot(epochs_range, val_loss2, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

predictions2 = model2.predict(test)
prediction_classes2 = [class_names[np.argmax(pred)] for pred in predictions2]

errors2 = [not pred==label for pred,label in zip(prediction_classes2,true_labels)]
errors_index2 = [i for i in range(len(errors2)) if errors2[i]]

plt.figure(figsize=(10,10))
for i in errors_index2[0:5]:
    plt.imshow(list(test_unbatched)[i][0].numpy().astype(int))
    plt.title(class_names[list(test_unbatched)[i][1]])
    plt.xlabel('prediction: '+prediction_classes2[i])
    plt.show()