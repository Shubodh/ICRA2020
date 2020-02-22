from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np


def read(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []
	LBL = []

	for line in A:
		(x, y, theta, lbl) = line.split(' ')
		X.append(float(x))
		Y.append(float(y))
		THETA.append(float(theta))

	return (X, Y, THETA)


if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA) = read(fileName)

	sample = 10
	X_meta = [X[i] for i in xrange(len(X)) if i%sample == 0]
	Y_meta = [Y[i] for i in xrange(len(Y)) if i%sample == 0]
	THETA_meta = [THETA[i] for i in xrange(len(THETA)) if i%sample == 0]

	plt.plot(X_meta, Y_meta, 'bo')
	plt.plot(X, Y, 'k')

	for i in xrange(len(X_meta)):
		x2 = math.cos(THETA_meta[i]) + X_meta[i]
		y2 = math.sin(THETA_meta[i]) + Y_meta[i]
		plt.plot([X_meta[i], x2], [Y_meta[i], y2], 'r->')
	
	# plt.plot(X, Y, 'bo')
	# plt.plot(X_meta[17], Y_meta[17], 'ro')
	# plt.plot(X_meta[7], Y_meta[7], 'ro')
	# plt.plot(X_meta[39], Y_meta[39], 'ro')
	# plt.plot(X_meta[48], Y_meta[48], 'ro')
	# plt.xlim(-5, 25)
	# plt.ylim(-15, 15)
	plt.show()

