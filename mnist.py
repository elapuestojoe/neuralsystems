from sklearn.datasets.mldata import fetch_mldata
import numpy as np
mnist = fetch_mldata('mnist-original', data_home='./MNIST')
np.random.seed(1)
print(mnist.data.shape)

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
	train[i] = np.array(data[i])[:minDataSize//2]
	test[i] = np.array(data[i])[minDataSize//2+1:]

	print("Train {}: {}".format(i, train[i].shape))
	print("Test {}: {}".format(i, test[i].shape))

print(train[1][1].shape)

class Network():
	def __init__(self, architechture):
		self.num_layers = len(architechture)
		self.sizes = architechture
		self.biases = [np.random.randn(y, 1) for y in architechture[1:]]
		self.weights = [np.random.randn(y, x) for x, y in zip(architechture[:-1], architechture[1:])]

		print(self.biases)
		print(self.weights)
		print(self.num_layers)

n = Network([3,2,2])
