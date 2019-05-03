import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras import metrics
from keras.models import Model
from keras.layers import Input, Dense
from keras import regularizers
import os
import gzip
import numpy as np
import pickle
# from tempfile import TemporaryFile

DELTA = 127

def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

def load_mnist(path, kind='train'):
	labels_path = os.path.join(path,
							   '%s-labels-idx1-ubyte.gz'
							   % kind)
	images_path = os.path.join(path,
							   '%s-images-idx3-ubyte.gz'
							   % kind)

	with gzip.open(labels_path, 'rb') as lbpath:
		labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
							   offset=8)

	with gzip.open(images_path, 'rb') as imgpath:
		images = np.frombuffer(imgpath.read(), dtype=np.uint8,
							   offset=16).reshape(len(labels), 784)

	return images, labels

def binarization(delta,array):
	return (np.where(array>delta, 1, 0))

def load_data():
	x_train, y_train = load_mnist('p2_data/mnist/', kind='train')
	x_test, y_test = load_mnist('p2_data/mnist/', kind='t10k')
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
	print (x_train.shape)
	print (x_test.shape)
	return x_train,x_test

def autoencoder(x_train,x_test,func):
	input_img = Input(shape=(784,))
	encoded = Dense(256, activation=func)(input_img)
	encoded = Dense(128, activation=func)(encoded)
	encoded = Dense(64, activation=func)(encoded)
	decoded = Dense(128, activation=func)(encoded)
	decoded = Dense(256, activation=func)(decoded)
	decoded = Dense(784, activation=func)(decoded)
	encoder = Model(input_img, encoded)
	autoencoder = Model(input_img, decoded)
	autoencoder.summary()
	autoencoder.compile(optimizer='adadelta',loss='mse',metrics=['mean_squared_error'])
	autoencoder.fit(x_train,x_train,epochs=25,batch_size=256,validation_data=(x_test,x_test))
	print (autoencoder.metrics_names)
	print (autoencoder.evaluate(x_train,x_train))
	print (autoencoder.evaluate(x_test,x_test))
	encoded_images_data = encoder.predict(x_train)
	pickleUnload("p2_data/encoded_data_train.pkl",encoded_images_data)
	encoded_images = encoder.predict(x_test)
	pickleUnload("p2_data/encoded_data_test.pkl",encoded_images)
	decoded_images = autoencoder.predict(x_test)
	e3 = np.array(autoencoder.get_weights())
	np.save(func,e3)
	return encoded_images,decoded_images


def visualize(X_test,decoded_images,encoded_imgs):
	plt.figure(figsize=(40, 4))
	for i in range(10):
		ax = plt.subplot(3, 20, i + 1)
		plt.imshow(X_test[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# ax = plt.subplot(3, 20, i + 1 + 20)
		# plt.imshow(encoded_imgs[i].reshape(8,8))
		# plt.gray()
		# ax.get_xaxis().set_visible(False)
		# ax.get_yaxis().set_visible(False)

		ax = plt.subplot(3, 20, i + 1 + 20)
		plt.imshow(decoded_images[i].reshape(28, 28))
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()

x_train,x_test = load_data()
encoded_images,decoded_images = autoencoder(x_train,x_test,'relu')
visualize(x_test,decoded_images,encoded_images)
