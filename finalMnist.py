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
		# print("loading previous net with weights ", n.weights)
		print("Loading previous net with NIterations", n.NIterations)
else:
	n = Network([1, 10, 300, 300, 300, 784])

# random.seed(5)
batchSize = 1
numberOfBatches = 1000

# Train
for batch in range(numberOfBatches):
	for i in range(batchSize):
		j = None
		objective = None
		result = None

		j = random.randint(0, len(x_train) - 1)
		objective = y_train[j]

		result = n.feedForward(x_train[j])

		n.backPropagate(result, [objective])
	n.updateWeights()

#Test code
stats = np.array([0,0,0])

# Matriz de confusion
yResults = []
predictedResults = []

for i in range(1000):
	j = random.randint(0, len(x_test) -1)
	objective = y_test[j]

	result = n.feedForward(x_test[j])
	tempResult = result[0]

	if(tempResult >= 0.5):
		tempResult = 1.0
	else:
		tempResult = 0.0

	yResults.append(y_test[j])
	predictedResults.append(tempResult)
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

# Plot confusion matrix
score = stats[0]/sum(stats)
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