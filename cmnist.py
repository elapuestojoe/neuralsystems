from sklearn.datasets.mldata import fetch_mldata
import numpy as np 
import matplotlib.pyplot as plt

# if __name__ == '__main__':
# 	train()


mnist = fetch_mldata('mnist-original', data_home='./MNIST')

for i in range(len(mnist.target)):
	if(mnist.target[i]) == 5:
		mnist.target[i] = 1
	else:
		mnist.target[i] = 0
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.95, random_state=0)

from network import Network
import random
import pickle
from pathlib import Path 

neuralNetFile = Path("./neuralNetwork.pkl")
n = None
if neuralNetFile.is_file():
	with open("./neuralNetwork.pkl", "rb") as inputF:
		n = pickle.load(inputF)
		print("loading previous net with weights ", n.weights)
else:
	n = Network([1, 20, 40, 50, 784])

# Batches
batchSize = 1
numberOfBatches = 10
print("Batch Size: ", batchSize)
print("Number of Batches: ", numberOfBatches)

indexTestOverFit = []

for batch in range(numberOfBatches):
	print("Batch: ", batch)
	for i in range(batchSize):
		j = random.randint(0, len(x_train) - 1)
		objective = y_train[j]
		indexTestOverFit.append([j,objective])

		result = n.feedForward(x_train[j])

		n.backPropagate(result, [objective])

	n.updateWeights()

# Save network
with open("neuralNetwork.pkl", "wb") as output:
	pickle.dump(n, output, pickle.HIGHEST_PROTOCOL)

print("testing")
# # Test
fails = 0
# testLenght = 100
testLenght = len(indexTestOverFit)

for index in indexTestOverFit:
	objective = index[1]
	j = index[0]
	result = n.feedForward(x_train[j])[0]

	if(result >= 0.5):
		result = 1
	else:
		result = 0

	if(result != objective):
		print("expected {}, got {}".format(objective, result))
		fails += 1

print("Fails = {}/{}".format(fails, testLenght))