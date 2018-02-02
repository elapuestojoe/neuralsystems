from sklearn.datasets.mldata import fetch_mldata
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from pathlib import Path 
from bnetwork import Network
import pickle
import random
import numpy as np 

# 91.9 on 
# get data 784
mnist = fetch_mldata('mnist-original', data_home='./MNIST')

# convert to what we are going to predict

filename = "./neuralNetworkST.pkl"

for i in range(len(mnist.target)):
	if(mnist.target[i]) == 5:
		mnist.target[i] = 0.9
	else:
		mnist.target[i] = 0.1

# normalize
mnist.data = normalize(mnist.data)

# divide into train/test
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.10, random_state=0)

#get array of fives
fiveDataset = {"data":[], "target":[]}
for i in range(len(y_train)):
	if(y_train[i]) == 0.9:
		fiveDataset["data"].append(x_train[i])
		fiveDataset["target"].append(y_train[i])

# load neural net or create it
neuralNetFile = Path(filename)
n = None
if neuralNetFile.is_file():
	with open(filename, "rb") as inputF:
		n = pickle.load(inputF)
		# print("loading previous net with weights ", n.weights)
		print("Loading previous net with NIterations", n.NIterations)
else:
	n = Network([1, 10, 300, 300, 300, 784])

# random.seed(5)
batchSize = 1
numberOfBatches = 5000
# Train
for batch in range(numberOfBatches):
	for i in range(batchSize):
		j = None
		objective = None
		result = None

		if(random.random() >= .8):
			j = random.randint(0, len(fiveDataset["target"])-1)
			objective = fiveDataset["target"][j]

			result = n.feedForward(fiveDataset["data"][j])[0]
		else:
			j = random.randint(0, len(x_train) - 1)
			objective = y_train[j]

			result = n.feedForward(x_train[j])[0]

		n.backPropagate(result, [objective])
	n.updateWeights()

# Matriz de confusion
yResults = []
predictedResults = []
size = 1000
correct = 0
for i in range(size):

	j = random.randint(0, len(x_test) -1)
	objective = y_test[j]

	result = n.feedForward(x_test[j])
	tempResult = result[0]

	if(objective == 0.9):
		objective = 1
	else:
		objective = 0

	if(tempResult >= 0.5):
		tempResult = 1
	else:
		tempResult = 0

	yResults.append(objective)
	predictedResults.append(tempResult)

	if(tempResult == objective):
		correct += 1

# Display results
print("Correct {}".format(correct/size))

print("Total", size)
# Save network
with open("neuralNetworkImproved.pkl", "wb") as output:
	pickle.dump(n, output, pickle.HIGHEST_PROTOCOL)

# Plot confusion matrix
score = correct/size #correctas / totales
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(yResults, predictedResults)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()