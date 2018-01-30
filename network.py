import numpy as np
import random
from math_utilities import sigmoid, derivative_sigmoid
sizeOfInput = 784
np.random.seed(1)
random.seed(1)

class Network():

	def createDeltaArray(self):
		deltas = []
		for i in range(0, len(self.layers)-1):
			deltaLayer = [0]*self.layers[i]
			deltas.append(deltaLayer)
		return deltas

	def resetOutputArray(self):
		self.outputs = []
		for i in range(0, self.num_layers-1):
			self.outputs.append([0]*self.layers[i])

	def __init__(self, architechture, learningRate=0.85, classification=True):
		self.num_layers = len(architechture)
		self.layers = architechture
		self.biases = [-1 for i in architechture[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(architechture[:-1], architechture[1:])]
		self.learningRate = learningRate

		self.biasWeights = []
		for i in range(0, len(self.layers)-1):
			self.biasWeights.append([random.random() for i in range(self.layers[i])])

		self.classification = classification

		self.deltas = self.createDeltaArray()
		self.resetOutputArray()
		# self.deltas
		# print("BIAS: ", self.biases)
		# print("biasWeights", self.biasWeights)
		# print("Weights: ", self.weights)
		print("Number of layers: ", self.num_layers)
		print("Deltas: ", self.deltas)


	
		print("Outputs: ", self.outputs)
	def feedForward(self, inputArray):
		self.resetOutputArray()
		for layer in range(self.num_layers-2, -1, -1):
			for i in range(len(inputArray)):
				for j in range(len(self.weights[layer][i])):
					self.outputs[layer][j] += (inputArray[i] * self.weights[layer][i][j])

			# apply bias:
			for i in range(len(self.outputs[layer])):
				self.outputs[layer][i] += self.biases[layer] * self.biasWeights[layer][i]

				# apply sigmoid
				self.outputs[layer][i] = sigmoid(self.outputs[layer][i])
			inputArray = self.outputs[layer]
		# Handle activation on classification problems
		# if(self.classification):
		# 	for i in range(len(self.outputs[0])):
		# 		if(self.outputs[0][i] >= 0.5):
		# 			self.outputs[0][i] = 1
		# 		else:
		# 			self.outputs[0][i] = 0

		result = self.outputs[0]
		return result

	def accumulateDeltas(self, deltaArray):
		for i in range(len(deltaArray)):
			for j in range(len(deltaArray[i])):
				self.deltas[i][j] += deltaArray[i][j]
				
	def backPropagate(self, output, expectedOutput):
		tempDeltas = self.createDeltaArray()
		for i in range(len(output)):
			tempDeltas[0][i] += derivative_sigmoid(output[i]) * (expectedOutput[i] - output[i])

		# propagate to hidden layers
		for i in range(1, self.num_layers-1):
			for j in range(0, len(self.outputs[i])):
				output = self.outputs[i][j]
				for delta in tempDeltas[i-1]:
					for weight in self.weights[i]:
						tempDeltas[i][j] +=  delta * weight[j]

				tempDeltas[i][j] *= derivative_sigmoid(output)
		self.accumulateDeltas(tempDeltas)
		# print("BackPropagate", self.deltas)

	def updateWeights(self):
		# print("updateWeights")

		# print("deltas: ", self.deltas)
		# print("weights", self.weights)
		for i in range(len(self.weights)):
			for j in range(len(self.weights[i])):
				for k in range(len(self.weights[i][j])):
					self.weights[i][j][k] += (self.learningRate * self.deltas[i][k]) #* 10 # * self.outputs[i][k]
		# update bias weights
		for i in range(len(self.biasWeights)):
			for j in range(len(self.biasWeights[i])):
				self.biasWeights[i][j] += self.learningRate * self.deltas[i][j] #* 10 #* self.biases[i]

		self.deltas = self.createDeltaArray()

# n = Network([1,400,400,800])
# n.feedForward([random.random() for i in range(800)])

# print(n.feedForward([1,2,2]))
# print(n.feedForward([1,2,2]))
# n.backPropagate([0.4],[0.8])
# n.backPropagate([0.4],[0.8])
# n.updateWeights()
# n.updateWeights()

n = Network([1,400,400,800])
r = n.feedForward([random.random() for i in range(800)])[0]
n.backPropagate([r], [random.random() for i in range(800)])