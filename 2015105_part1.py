import pickle as cPickle
import gzip
import numpy as np
import random

def pickleLoad(filename):
	with open(filename, "rb") as f:
		filetype = cPickle.load(f)
	return filetype

def pickleUnload(filename,filetype):
	with open(filename, "wb") as f:
		cPickle.dump(filetype, f)

class Network(object):
	def __init__(self, sizes, lr, epochs, batch_size, activation_function):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.learning_rate = lr
		self.epochs = epochs
		self.batch_size = batch_size
		self.activation = activation_function
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_deriv(self,z):
		return self.sigmoid(z)*(1-self.sigmoid(z))

	def relu(self,z):
		return np.maximum(0,z)

	def relu_deriv(self,z):
		dZ = np.array(z, copy=True)
		dZ[z<0] = 0
		return dZ

	def linear(self,z):
		return z

	def linear_deriv(self,z):
		return np.array(z, copy=True)

	def tanh(self,z):
		ez = np.exp(z)
		enz = np.exp(-z)
		return (ez - enz)/ (ez + enz)

	def tanh_deriv(self,z):
		a = tanh(z)
		dz = 1 - a**2
		return dz

	def softmax(self,z):
		return np.exp(z) / np.sum(np.exp(z), axis =0)

	def feedforward(self, a):
		for b, w in zip(self.biases, self.weights):
			if self.activation == 'relu':
				a = self.relu(np.dot(w, a)+b)
			elif self.activation == 'sigmoid':
				a = self.sigmoid(np.dot(w, a)+b)
			elif self.activation == 'linear':
				a = self.linear(np.dot(w, a)+b)
			elif self.activation == 'tanh':
				a = self.tanh(np.dot(w, a)+b)
		return a

	def fit(self, training_data, test_data=None):
		for j in range(self.epochs):
			random.shuffle(training_data)
			batches = [training_data[k:k+self.batch_size] for k in range(0, len(training_data), self.batch_size)]
			for batch in batches:

			print("Epoch {0} complete".format(j))

	def update_layer(self, batch):
		layer_bias = [np.zeros(b.shape) for b in self.biases]
		layer_weight = [np.zeros(w.shape) for w in self.weights]
		for x, y in batch:
			delta_layer_bias, delta_layer_weight = self.backpropagation(x, y)
			layer_weight = [nw+dnw for nw, dnw in zip(layer_weight, delta_layer_weight)]
			layer_bias = [nb+dnb for nb, dnb in zip(layer_bias, delta_layer_bias)]
		self.biases = [b-(self.learning_rate/len(batch))*nb for b, nb in zip(self.biases, layer_bias)]
		self.weights = [w-(self.learning_rate/len(batch))*nw for w, nw in zip(self.weights, layer_weight)]

	def backpropagation(self, x, y):
		zs = []
		activation = x
		activations = [x]
		layer_bias = [np.zeros(b.shape) for b in self.biases]
		layer_weight = [np.zeros(w.shape) for w in self.weights]
		for b, w in zip(self.biases, self.weights):
			z = np.dot(w, activation)+b
			zs.append(z)
			if self.activation == 'relu':
				activation = self.relu(z)
			elif self.activation == 'sigmoid':
				activation = self.sigmoid(z)
			elif self.activation == 'linear':
				activation = self.linear(z)
			elif self.activation == 'tanh':
				activation = self.tanh(z)
			activations.append(activation)

		if self.activation == 'relu':
			delta = (activations[-1] - y) * self.relu_deriv(zs[-1])
		elif self.activation == 'sigmoid':
			delta = (activations[-1] - y) * self.sigmoid_deriv(zs[-1])
		elif self.activation == 'linear':
			delta = (activations[-1] - y) * self.linear_deriv(zs[-1])
		elif self.activation == 'tanh':
			delta = (activations[-1] - y) * self.tanh_deriv(zs[-1])
		layer_bias[-1] = delta
		layer_weight[-1] = np.dot(delta, activations[-2].transpose())
		for l in range(2, self.num_layers):
			z = zs[-l]
			if self.activation == 'relu':
				sp = self.relu_deriv(z)
			elif self.activation == 'sigmoid':
				sp = self.sigmoid_deriv(z)
			elif self.activation == 'linear':
				sp = self.linear_deriv(z)
			elif self.activation == 'tanh':
				sp = self.tanh_deriv(z)
			delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
			layer_bias[-l] = delta
			layer_weight[-l] = np.dot(delta, activations[-l-1].transpose())
		return (layer_bias, layer_weight)

	def predict(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		predictions = [int(x) for (x, y) in test_results]
		return predictions

	def score(self, test_data):
		test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		correct_count = sum(int(x == y) for (x, y) in test_results)
		return correct_count/len(test_data)


training_data = pickleLoad("p1_data/mnist_training_data.pkl")
validation_data = pickleLoad("p1_data/mnist_validation_data.pkl")
test_data = pickleLoad("p1_data/mnist_test_data.pkl")

net = Network([784, 256, 128, 64, 10],0.1,100,256,'relu')
net.fit(training_data,test_data=validation_data)
print (net.score(test_data))
