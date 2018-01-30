from sklearn.datasets.mldata import fetch_mldata
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from pathlib import Path 
from bnetwork import Network
import pickle
import random
import numpy as np 

# get data
mnist = fetch_mldata('mnist-original', data_home='./MNIST')

# convert to what we are going to predict

fiveDataset = {"data":[], "target":[]}

for i in range(len(mnist.target)):
	if(mnist.target[i]) == 5:
		mnist.target[i] = 1
		fiveDataset["data"].append(mnist.data[i])
		fiveDataset["target"].append(mnist.target[i])

	else:
		mnist.target[i] = 0

# normalize
mnist.data = normalize(mnist.data)

# divide into train/test
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.10, random_state=0)

# load neural net or create it
neuralNetFile = Path("./neuralNetworkImproved.pkl")
n = None
if neuralNetFile.is_file():
	with open("./neuralNetworkImproved.pkl", "rb") as inputF:
		n = pickle.load(inputF)
		print("loading previous net with weights ", n.weights)
else:
	n = Network([1, 10, 300, 300, 300, 784])

# random.seed(5)
batchSize = 1
numberOfBatches = 1000

# Train
# stats = np.array([0,0,0])
for batch in range(numberOfBatches):
	# print("Batch: ", batch)
	for i in range(batchSize):
		j = None
		objective = None
		result = None
		# train with 50% 5 50% other
		if(random.random()<=0.5):
			j = random.randint(0, len(fiveDataset["data"])-1)
			objective = fiveDataset["target"][j]
			result = n.feedForward(fiveDataset["data"][j])

		else:
			j = random.randint(0, len(x_train) - 1)
			objective = y_train[j]

			result = n.feedForward(x_train[j])

		n.backPropagate(result, [objective])

		# tempResult = result[0]
		# if(tempResult >= 0.5):
			# tempResult = 1.0
		# else:
			# tempResult = 0.0
		# print("Expected {}, got {}".format(objective, tempResult))
		# stats[int(objective) - int(tempResult)] += 1
	n.updateWeights()

stats = np.array([0,0,0])
for i in range(100):
	j = random.randint(0, len(x_test) -1)
	objective = y_test[j]

	result = n.feedForward(x_test[j])
	tempResult = result[0]

	if(tempResult >= 0.5):
		tempResult = 1.0
	else:
		tempResult = 0.0
	# print("Expected {}, got {}".format(objective, tempResult))
	stats[int(objective) - int(tempResult)] += 1
# Display results
print(stats)

print("Correct {}".format(stats[0]/sum(stats)))
print("False positives {}".format(stats[-1]/sum(stats)))
print("False negatives {}".format(stats[1]/sum(stats)))
print("Total", sum(stats))
# Save network
with open("neuralNetworkImproved.pkl", "wb") as output:
	pickle.dump(n, output, pickle.HIGHEST_PROTOCOL)