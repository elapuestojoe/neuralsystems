import math
import random

def sigmoid(h):
	return (1/(1+math.exp(-h)))

def dSigmoid(h):
	return (x*(1 - x))

class Node():
	def __init__(self):
		self.weight = random.random()
		self.value = 0
	def __repr__(self):
		return str("W: {}, V: {}".format(self.weight, self.value))

class Layer():

	def __init__(self, numberOfNodes, bias=-1):
		self.nodes = []
		for i in range(numberOfNodes):
			self.nodes.append(Node())
		self.bias = bias
		self.biasWeight = random.random()

	def __repr__(self):
		return ", ".join(map(repr, self.nodes))+ " Bias: {}, biasWeight: {}\n".format(self.bias, self.biasWeight)

class NN():

	learningParameter = 0.5

	layers = []
	inputLayer = None
	
	def addLayer(self, n):
		self.layers.append(Layer(n))

	def feedForward(self, inputVector):
		if(len(inputVector) == len(self.layers[0].nodes)):
			
			self.layers[0].bias = 1
			for i in range(len(self.layers[0].nodes)):
				self.layers[0].nodes[i].value = inputVector[i]

			for i in range(1, len(self.layers)):
				layer = self.layers[i]
				previousLayer = self.layers[i-1]

				for j in range(len(layer.nodes)):
					node = layer.nodes[j]
					for k in range(len(previousLayer.nodes)):
						node.value += previousLayer.nodes[k].value * previousLayer.nodes[k].weight

		else:
			print("Expected inputVector of size {}".format(len(self.layers[0].nodes)))

	def __repr__(self):
		text = ""
		for layer in self.layers:
			text+=repr(layer)
		return text
nn = NN()
nn.addLayer(2)
nn.addLayer(2)
nn.addLayer(2)

nn.feedForward([4,1])

print(nn)