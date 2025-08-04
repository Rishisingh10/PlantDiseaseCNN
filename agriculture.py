# Imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPool2D, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import json

# Data Preprocessing
training_set = tf.keras.utils.image_dataset_from_directory(
    'D:/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    verbose=True,
)

validation_set = tf.keras.utils.image_dataset_from_directory(
    'D:/dataset/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid',
    labels="inferred",
    label_mode="categorical",
    batch_size=32,
    image_size=(128, 128),
    shuffle=True,
    verbose=True,
)

# Model Building
model = Sequential([
Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(128, 128, 3)),
    Conv2D(32, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(64, kernel_size=3, padding='same', activation='relu'),
    Conv2D(64, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(128, kernel_size=3, padding='same', activation='relu'),
    Conv2D(128, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(256, kernel_size=3, padding='same', activation='relu'),
    Conv2D(256, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Conv2D(512, kernel_size=3, padding='same', activation='relu'),
    Conv2D(512, kernel_size=3, activation='relu'),
    MaxPool2D(pool_size=2, strides=2),

    Dropout(0.25),
    Flatten(),
    Dense(1500, activation='relu'),
    Dropout(0.4),
    Dense(38, activation='softmax')
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Checkpoint
checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)

# Train Model
training_history = model.fit(
    training_set,
    validation_data=validation_set,
    epochs=10,
    callbacks=[checkpoint]
)
#Model Evaluation
train_loss,train_acc=model.evaluate(training_set)
print('Training Loss:',train_loss)
print('Training Accuracy:',train_acc)
validation_loss,validation_acc=model.evaluate(validation_set)
print('Validation Loss:',validation_loss)
print('Validation Accuracy:',validation_acc)
model.save('best_model.h5')
#Recording history in JSON
import json
with open('training_history.json', 'w') as f:
    json.dump(training_history.history, f)
#Accuracy Visualization
epochs= [i for i in range(1, 11)]
plt.plot(epochs, training_history.history['accuracy'],color='red',label='Training accuracy')
plt.show()
epochs= [i for i in range(1, 11)]
plt.plot(epochs, training_history.history['accuracy'],color='blue',label='Validation accuracy')
plt.show()
