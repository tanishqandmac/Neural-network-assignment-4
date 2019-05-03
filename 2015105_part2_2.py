import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import keras
import tensorflow as tf
from keras.layers import LSTM,SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from collections import Counter
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scikitplot.plotters as skplt
import pickle
from utils_2015105 import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import gzip

top_words = 6000
epoch_num = 5
batch_size = 64

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

def plot_cmat_seaborn(y_test, y_pred,name):
	import seaborn as sn
	from sklearn.metrics import confusion_matrix
	conf_mat = confusion_matrix(y_test, y_pred)
	conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
	sn.set(font_scale=1.4)
	sn.heatmap(conf_mat, fmt='g', annot=True,annot_kws={"size": 16})# font size
	title = "Confusion Matrix: "+name
	plt.title(title)
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = pickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		pickle.dump(filetype, f)

def plot_cmat(yte, ypred):
    skplt.plot_confusion_matrix(yte, ypred)
    plt.show()

def plot_cmat_seaborn(y_test, y_pred):
    import seaborn as sn
    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)
    conf_mat_normalized = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    sn.set(font_scale=1.4)
    sn.heatmap(conf_mat, fmt='g', annot=True,annot_kws={"size": 16})# font size
    plt.title("Confusion Matrix: Neural Network")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def load_data():
	x_train, y_train = load_mnist('p2_data/mnist/', kind='train')
	x_test, y_test = load_mnist('p2_data/mnist/', kind='t10k')
	x_train = x_train.astype('float32')/255.0
	x_test = x_test.astype('float32')/255.0
	x_train = x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))
	print (x_train.shape)
	print (x_test.shape)
	return x_train,x_test,y_train,y_test

X_train,X_test,y_train,y_test = load_data()
X_train = pickleLoad("p2_data/encoded_data_train.pkl")
X_test = pickleLoad("p2_data/encoded_data_test.pkl")
y_test1 = y_test
y_train = np.array(to_categorical(y_train))
y_test = np.array(to_categorical(y_test))

model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=64)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.predict_classes(X_test)
scores1 = model.evaluate(X_test, y_test)
predictions = [int(a) for a in scores]
plot_cmat_seaborn(y_test1,predictions)


#Bonus part
pca = PCA(n_components=64).fit(X_train)
X_train = pca.transform(X_train)
X_test = pca.transform(X_test)
model = Sequential()
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=64)
scores = model.evaluate(X_test, y_test)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
scores = model.predict_classes(X_test)
scores1 = model.evaluate(X_test, y_test)
predictions = [int(a) for a in scores]
plot_cmat_seaborn(y_test1,predictions)
