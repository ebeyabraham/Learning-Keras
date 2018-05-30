"""
Linear regression on the boston housing dataset to predict the cost of a house
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import boston_housing

(X_train, Y_train), (X_test, Y_test) = boston_housing.load_data()

nFeatures = X_train.shape[1]

model = Sequential()
model.add(Dense(1, input_shape = (nFeatures,), activation = "linear"))

model.compile(loss = "mse", optimizer = "rmsprop", metrics = ["accuracy"])

model.fit(X_train, Y_train, epochs = 1000, batch_size = 4)

model.evaluate(X_test, Y_test, verbose=True)
Y_pred = model.predict(X_test)

#Compare ground truth with predictions
print(Y_test[:5])
print(Y_pred[:5,0])