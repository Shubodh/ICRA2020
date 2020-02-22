import matplotlib.pyplot as plt
from sys import argv, exit
import math


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

	# X_temp = X
	# Y_temp = Y
	# X = [y for y in Y_temp]
	# Y = [-x for x in X_temp]

	return (X, Y, THETA)


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

	# X_temp = X
	# Y_temp = Y
	# X = [-y for y in Y_temp]
	# Y = [x for x in X_temp]

	return (X, Y, THETA, LBL)


def readKitti(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	X = []
	Y = []
	THETA = []

	for line in A:
		l = line.split(' ')
		
		x = float(l[3]); y = float(l[7]); theta = math.atan2(float(l[4]), float(l[0]))
		
		X.append(x)
		Y.append(y)
		THETA.append(theta)

	return (X, Y, THETA)


def draw(X1, Y1, X2, Y2, gt=True, g2o=True):
	width = 2.5

	if(gt == True):
		plt.plot(X1, Y1, 'b-', markersize=5, linewidth=width ,label='Ground Truth')

	if(g2o == True):
		plt.plot(X2, Y2, 'g-', markersize=5, linewidth=width, label='RTABMAP G2O')

	plt.legend()
	plt.axis('scaled')
	plt.show()


if __name__ == '__main__':
	fileGt = str(argv[1])
	fileG2o = str(argv[2])

	# (X1, Y1, THETA1, LBL1) = readTxt(fileGt)
	# (X2, Y2, THETA2) = readG2o(fileG2o)
	(X1, Y1, THETA1) = readKitti(fileGt)
	# X1 = X1[4:]; Y1 = Y1[4:]; THETA1 = THETA1[4:]
	(X2, Y2, THETA2) = readKitti(fileG2o)

	draw(X1, Y1, X2, Y2, gt=True, g2o=True)