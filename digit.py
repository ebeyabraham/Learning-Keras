"""
Handwritten Digit Classification Using Feed Forward Networks
"""
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("Training data shape : ", train_images.shape, train_labels.shape)
print("Testing data shapes: ", test_images.shape, test_labels.shape)

classes = np.unique(train_labels)
nClasses = len(classes)

#reshape the data from 2D to a linear form
dimData = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dimData)
test_data = test_images.reshape(test_images.shape[0], dimData)

#convert to float
train_data = train_data.astype("float32")
test_data = test_data.astype("float32")

#change the scale of grayscale from 0-255 to 0-1
train_data /= 255
test_data /= 255

#categorize the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model = Sequential()
model.add(Dense(512, input_shape = (dimData,), activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(512, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(nClasses, activation = "softmax"))

model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(train_data, train_labels, batch_size = 200, epochs = 20, verbose = 1, validation_data=(test_data, test_labels))

[test_loss, test_acc] = model.evaluate(test_data, test_labels)
print("Loss = {} Accuracy = {}".format(test_loss, test_acc))

print(history.history.keys())

#Loss Curves
plt.figure()
plt.plot(history.history['loss'], 'r', linewidth = 3)
plt.plot(history.history['val_loss'], 'b', linewidth = 3)
plt.legend(["Training Loss", "Validation Loss"], fontsize = 18)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Loss", fontsize = 16)
plt.title("Loss Curves", fontsize = 16)

#Accuracy Curves
plt.figure()
plt.plot(history.history['acc'], 'r', linewidth = 3)
plt.plot(history.history['val_acc'], 'b', linewidth = 3)
plt.legend(["Training Accuracy", "Validation Accuracy"], fontsize = 18)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Accuracy", fontsize = 16)
plt.title("Accuracy Curves", fontsize = 16)

plt.show()
