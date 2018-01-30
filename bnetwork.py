import numpy as np
import random
from math_utilities import sigmoid, derivative_sigmoid

np.random.seed(1)
random.seed(1)

class Network():

	def createDeltaArray(self):
		return [np.zeros([1,x]) for x in self.layers[:-1]]

	def resetOutputArray(self):
		# self.outputArray = [np.zeros([1,x]) for x in layers[:-1]]
		pass

	def status(self):
		# print("weights", self.weights)
		# print("deltas", self.deltas)
		# print("biases", self.biases)
		# print("outputArray", self.outputArray)
		# print("biasWeights", self.biasWeights)
		print("")

	def __init__(self, layers, learningRate=0.5):
		
		self.num_layers = len(layers)

		self.layers = layers
		
		self.deltas = self.createDeltaArray()

		self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

		self.biases = np.empty(self.num_layers-1)
		self.biases.fill(-1)

		self.biasWeights = [np.random.randn(1,x) for x in layers[:-1]]

		self.learningRate = learningRate

		self.outputArray = [np.zeros([1,x]) for x in layers[:-1]]

		self.status()

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
		tempDeltas = self.createDeltaArray()

		output = np.array(output)
		expectedOutput = np.array(expectedOutput)

		tempDeltas[0] = (np.apply_along_axis(derivative_sigmoid, 0, output.reshape(1,len(output)))) * (expectedOutput - output)

		# propagate to hidden layers
		for i in range(1, self.num_layers-1):
			tempDeltas[i] = (np.apply_along_axis(derivative_sigmoid, 0, self.outputArray[i])) * (np.matmul(tempDeltas[i-1], np.transpose(self.weights[i-1])))

		# accumulate deltas
		for i in range(len(self.deltas)):
			self.deltas[i] = np.add(self.deltas[i], tempDeltas[i])

	def updateWeights(self):
		# print("updateWeights")

		for layer in range(len(self.weights)):
			# Todo: check input 
			self.weights[layer] +=  (self.deltas[layer] *(self.learningRate))
			self.biasWeights[layer] += (self.deltas[layer] * (self.learningRate))

		self.deltas = self.createDeltaArray()

# n = Network([1, 10, 30, 300, 784])

# for layer in n.weights:
# 	print(layer.shape)

# # print(n.feedForward([1,1,2]))
# n.feedForward([1,2,2])
# n.backPropagate([0.5],[0.1])
# n.backPropagate([0.5],[0.1])
# n.feedForward([1,2,2])
# n.backPropagate([0.5],[10])
# print(n.deltas)
# n.updateWeights()

# n.feedForward([1,2,2])

# # n.backPropagate([0.5],[0.1])
# # n.backPropagate([0.5],[0.1])
# n.backPropagate([0.5],[0.1])

# n.updateWeights()
