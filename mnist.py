from sklearn.datasets.mldata import fetch_mldata
import numpy as np
from sklearn.preprocessing import normalize

mnist = fetch_mldata('mnist-original', data_home='./MNIST')
# np.random.seed(1)
print(mnist.data.shape)

# normalize data
mnist.data = normalize(mnist.data)
X = mnist.target.shape
print(X)
y = np.unique(mnist.target)
print(y)

#print(mnist.data[0].shape) #28x28 pixeles 0-255 #784
# nodes = np.random.rand(784,1)
# net = np.array([[1],nodes,np.ones(784)])
# y = np.array([[0,1,0]]).T
# print(nodes)

# Divide into train/test
data = []
test = []
train = []
for i in range(len(np.unique(mnist.target))):
	data.append([])
	test.append([])
	train.append([])

for i in range(len(mnist.target)):
	data[int(mnist.target[i])].append(mnist.data[i])

minDataSize = len(data[0])
minDataIndex = 0
for i in range(1, len(data)):
	if len(data[i]) < minDataSize:
		minDataSize = len(data[i])
		minDataIndex = i

for i in range(len(data)):
	# Divide the data into 50% train 50% test
	train[i] = np.array(data[i])[:minDataSize//10]
	test[i] = np.array(data[i])[minDataSize//10+1:]

	print("Train {}: {}".format(i, train[i].shape))
	print("Test {}: {}".format(i, test[i].shape))

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
	n = Network([1, 50, 50, 50, 50, 50, 784])

# Batches
batchSize = 1
numberOfBatches = 10
print("Batch Size: ", batchSize)
print("Number of Batches: ", numberOfBatches)

indexTestOverFit = []
for batch in range(numberOfBatches):
	print("Batch: ", batch)
	for i in range(batchSize):
		j = random.randint(4, 5)
		k = random.randint(0, len(train[j])-1)
		indexTestOverFit.append([j,k])
		expectedResult = 0
		if(j==5):
			expectedResult = 1

		result = n.feedForward(train[j][k])

		n.backPropagate(result, [expectedResult])

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
	j = index[0]
	k = index[1]
	expectedResult = 0
	if(j==5):
		expectedResult = 1
	result = n.feedForward(train[j][k])[0]
	if(result >= 0.5):
		result = 1
	else:
		result = 0
	if(result != expectedResult):
		fails += 1

# for i in range(testLenght):
# 	j = random.randint(0,9)
# 	k = random.randint(0, len(train[j]-1)) #temp
# 	expectedResult = 0
# 	if(j==5):
# 		expectedResult = 1
# 	result = n.feedForward(train[j][k])[0]
# 	if(result >= 0.5):
# 		result = 1
# 	else:
# 		result = 0
# 	if(result != expectedResult):
# 		fails += 1

print("Fails = {}/{}".format(fails, testLenght))