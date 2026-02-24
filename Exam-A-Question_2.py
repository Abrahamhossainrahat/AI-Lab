# -*- coding: utf-8 -*-
"""Assignment-5-MNIST_Odd_CNN.ipynb"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# -----------------------------
# Data Processing
# -----------------------------

(x_train_full, y_train_full), (x_test_full, y_test_full) = mnist.load_data()

# combine full dataset
X = np.concatenate([x_train_full, x_test_full])
y = np.concatenate([y_train_full, y_test_full])

# keep only odd digits
odd_digits = [1,3,5,7,9]
mask = np.isin(y, odd_digits)

X = X[mask]
y = y[mask]

# convert labels → 0 to 4
label_map = {1:0, 3:1, 5:2, 7:3, 9:4}
y = np.array([label_map[i] for i in y])

# normalization
X = X / 255.0

# reshape for CNN
X = X.reshape(-1,28,28,1)

# 85% train, 15% test
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# -----------------------------
# Model Build
# -----------------------------

model = Sequential()

model.add(Conv2D(32,(3,3), activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64,(3,3), activation='relu'))

model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(5, activation='softmax'))

# -----------------------------
# Compile
# -----------------------------

optimizer = Adagrad(learning_rate=0.003)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# Model Checkpoint
# -----------------------------

checkpoint = ModelCheckpoint("best_model.h5",
                             monitor='val_loss',
                             save_best_only=True,
                             mode='min',
                             verbose=1)

# -----------------------------
# Phase 1 : First 10 Epochs (Full Training)
# -----------------------------

history1 = model.fit(x_train, y_train,
                     epochs=10,
                     batch_size=32,
                     validation_split=0.15,
                     callbacks=[checkpoint])

# -----------------------------
# Freeze First 3 Conv Layers
# -----------------------------

for layer in model.layers:
    if isinstance(layer, Conv2D):
        layer.trainable = False

# recompile after freezing
model.compile(optimizer=Adagrad(learning_rate=0.003),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# Phase 2 : Next 20 Epochs
# -----------------------------

history2 = model.fit(x_train, y_train,
                     epochs=20,
                     batch_size=32,
                     validation_split=0.15,
                     callbacks=[checkpoint])

# -----------------------------
# Evaluation
# -----------------------------

model.load_weights("best_model.h5")

loss, accuracy = model.evaluate(x_test, y_test)
print("Total loss: ", loss)
print("Total accuracy: ", accuracy)

# -----------------------------
# Plot Loss
# -----------------------------

plt.plot(history1.history['loss'] + history2.history['loss'],
         label='Training Loss', color='green')
plt.plot(history1.history['val_loss'] + history2.history['val_loss'],
         label='Validation Loss', color='red')

plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# -----------------------------
# Plot Accuracy
# -----------------------------

plt.plot(history1.history['accuracy'] + history2.history['accuracy'],
         label='Training Accuracy', color='green')
plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'],
         label='Validation Accuracy', color='red')

plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# -----------------------------
# Prediction
# -----------------------------

class_names = ["1","3","5","7","9"]

predictions = model.predict(x_test)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test

plt.figure(figsize=(12,8))

for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_test[i].reshape(28,28), cmap='gray')
    plt.title(f"P:{class_names[y_pred[i]]}, T:{class_names[y_true[i]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()
