from sys import argv, exit
import math
import numpy as np


def readTxt(fileName):
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
		LBL.append(float(lbl.rstrip('\n')))

	return (X, Y, THETA)


def readG2o(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for line in A:
		if "VERTEX_SE2" in line:
			(ver, ind, x, y, theta) = line.split(' ')
			X.append(float(x))
			Y.append(float(y))
			THETA.append(float(theta.rstrip('\n')))

	X_temp = X
	Y_temp = Y
	X = [y for y in Y_temp]
	Y = [-x for x in X_temp]

	return (X, Y, THETA)


def convert(X, Y, THETA):
	A = np.zeros((len(X), 12))

	for i in range(len(X)):
		T = np.identity(4)
		T[0, 3] = X[i]
		T[1, 3] = Y[i]
		R = np.array([[math.cos(THETA[i]), -math.sin(THETA[i]), 0], [math.sin(THETA[i]), math.cos(THETA[i]), 0], [0, 0, 1]])
		T[0:3, 0:3] = R
		
		A[i] = T[0:3, :].reshape(1, 12)
		
	return A


def draw(X, Y, THETA):
	ax = plt.subplot(111)
	ax.plot(X, Y, 'ro')
	ax.plot(X, Y, 'k-')

	plt.show()


if __name__ == '__main__':
	# (X, Y, THETA) = readG2o(argv[1])

	(X, Y, THETA) = readTxt(argv[1])
	X = X[70:]; Y = Y[70:]; THETA = THETA[70:]
	# X = X[0:2800]; Y = Y[0:2800]; THETA = THETA[0:2800]
	# X = X[3100: 6000]; Y = Y[3100: 6000]; THETA = THETA[3100: 6000]

	A = convert(X, Y, THETA)

	np.savetxt("unopt.kitti", A, delimiter=' ')
