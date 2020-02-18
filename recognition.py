import pandas as pd
import matplotlib.pyplot as plt
import numpy
import keras
import os

from keras.datasets import mnist
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import layers

def plot_history(history):
	plt.style.use('ggplot')
	acc = history.history['accuracy']
	val_acc = history.history['val_accuracy']
	loss = history.history['loss']
	val_loss = history.history['val_loss']
	x = range(1, len(acc) + 1)

	plt.figure(figsize=(12, 5))
	plt.subplot(1, 2, 1)
	plt.plot(x, acc, 'b', label='Training acc')
	plt.plot(x, val_acc, 'r', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.legend()
	plt.subplot(1, 2, 2)
	plt.plot(x, loss, 'b', label='Training loss')
	plt.plot(x, val_loss, 'r', label='Validation loss')
	plt.title('Training and validation loss')
	plt.legend()
	plt.savefig('plot.png')

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

print('Setting up model...')

# found that this setup achieved the highest accuracy of ~99.2%
model = Sequential()
model.add(layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

os.makedirs('nnmodels2conv', exist_ok=True)
filepath = 'nnmodels2conv/model-{epoch:02d}-{val_accuracy:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print('Training...')

history = model.fit(x_train, y_train, 
		epochs=100,
		verbose=True,
		validation_data=(x_test, y_test),
		callbacks=callbacks_list,
		batch_size=10)

plot_history(history)