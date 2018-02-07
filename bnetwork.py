import numpy as np
import random
from math_utilities import sigmoid, derivative_sigmoid

# np.random.seed(1)
# random.seed(1)

class Network():

	def createDeltaArray(self):
		return [np.zeros([1,x]) for x in self.layers[:-1]]

	def __init__(self, layers, learningRate=0.5):
		
		self.NIterations = 0
		self.updates = 0
		self.i = 0
		self.num_layers = len(layers)

		self.layers = layers
		
		self.deltas = self.createDeltaArray()

		self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

		self.biases = np.empty(self.num_layers-1)
		self.biases.fill(-1)

		self.biasWeights = [np.random.randn(1,x) for x in layers[:-1]]

		self.learningRate = learningRate

		self.outputArray = [np.zeros([1,x]) for x in layers[:-1]]

	def feedForward(self, inputArray):
		self.outputArray = [np.zeros([1,x]) for x in self.layers[:-1]] 
		for layer in range(self.num_layers-2, -1, -1):
			inputArray = np.array(inputArray).reshape(self.layers[layer+1],1)
			self.outputArray[layer] += (inputArray * self.weights[layer]).sum(axis=0)
			
			# # apply bias
			self.outputArray[layer] += (self.biases[layer] * self.biasWeights[layer])
			
			# apply sigmoid
			self.outputArray[layer] = [np.apply_along_axis(sigmoid, 0, self.outputArray[layer])]
			
			inputArray = self.outputArray[layer]

		return self.outputArray[0]

	def backPropagate(self, output, expectedOutput):

		self.NIterations +=1
		self.i += 1
		tempDeltas = self.createDeltaArray()

		output = np.array(output)
		expectedOutput = np.array(expectedOutput)

		tempDeltas[0] += (np.apply_along_axis(derivative_sigmoid, 0, output.reshape(1,len(output)))) * (expectedOutput - output)

		# propagate to hidden layers
		for i in range(1, self.num_layers-1):
			tempDeltas[i] += (np.apply_along_axis(derivative_sigmoid, 0, self.outputArray[i])) * (np.matmul(tempDeltas[i-1], np.transpose(self.weights[i-1])))

		# accumulate deltas
		for i in range(len(self.deltas)):
			self.deltas[i] = np.add(self.deltas[i], tempDeltas[i])

	def updateWeights(self):
		self.updates +=1
		for i in range(len(self.deltas)):
			self.deltas[i] = np.divide(self.deltas[i], self.i)

		for layer in range(len(self.weights)):
			# Todo: check input 
			self.weights[layer] +=  self.learningRate * (self.deltas[layer] * self.outputArray[layer])
			self.biasWeights[layer] += self.learningRate * (self.deltas[layer] * self.biases[layer])

		self.deltas = self.createDeltaArray()
		self.i = 0

	def predict(self, inputs, outputs):
		predOutputs = []
		for i in inputs:
			predOutputs.append(self.feedForward(i))

		return predOutputs

	def stats(self):
		print("Iterations", self.NIterations)
		print("Updates", self.updates)