import math

def sigmoid(n):
	if(n < 0):
		return 1 - 1 / (1+math.exp(n))
	return (1/(1+math.exp(-n)))

def derivative_sigmoid(n):
	return (n*(1 - n))