from sklearn.datasets.mldata import fetch_mldata
import numpy as np 
import matplotlib.pyplot as plt

mnist = fetch_mldata('mnist-original', data_home='./MNIST')

# see digits
# plt.figure(figsize=(20,4))

# for index, (image,label) in enumerate(zip(mnist.data[0:10000:2000], mnist.target[0:10000:2000])):
# 	plt.subplot(1, 5, index + 1)
# 	plt.imshow(np.reshape(image, (28,28)), cmap=plt.cm.gray)
# 	plt.title("Training {}".format(label))

# plt.show()

for i in range(len(mnist.target)):
	if(mnist.target[i]) == 5:
		mnist.target[i] = 1
	else:
		mnist.target[i] = 0
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.85, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
print("fit")
logisticRegr.fit(x_train, y_train)
print("endFit")
# Returns a NumPy Array
# Predict for One Observation (image)
# logisticRegr.predict(x_test[0].reshape(1,-1))

# logisticRegr.predict(x_test[0:10])

print("pred")
predictions = logisticRegr.predict(x_test)
print("endPred")
# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)

import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, predictions)
plt.figure(figsize=(9,9))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title = 'Accuracy Score: {0}'.format(score)
plt.title(all_sample_title, size = 15);
plt.show()