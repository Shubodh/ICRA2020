import matplotlib.pyplot as plt
from sys import argv
import math
import numpy as np


def readPose(filename):
	f = open(filename, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for i, line in enumerate(A):
		if(i % 3 == 0):
			(x, y, theta) = line.split(' ')
			# print(x, y, theta.rstrip('\n'))
			X.append(float(x))
			Y.append(float(y))
			THETA.append(math.radians(float(theta.rstrip('\n'))))

	return X, Y, THETA


def draw(X, Y, THETA):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'ro')
	ax.plot(X, Y, 'k-')

	plt.show()


if __name__ == '__main__':
	X, Y, THETA = readPose(argv[1])
	print(len(X))
	draw(X, Y, THETA)