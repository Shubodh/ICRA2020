import matplotlib.pyplot as plt
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

	return (X, Y, THETA, LBL)


def draw(X1, Y1, X2, Y2):
	plt.plot(X1, Y1, 'r-', markersize=5, linewidth=2 ,label='Noisy')
	plt.plot(X2, Y2, 'b-', markersize=5, linewidth=2, label='Ground Truth')

	plt.legend()
	plt.show()


def rectify(X1, Y1, THETA1, X2, Y2, THETA2):
	p1 = (X1[0], Y1[0], THETA1[0])
	p2 = (X2[0], Y2[0], THETA2[0])
	# print(p1, p2)

	Tw_n = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
	Tw_g = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])

	Tg_n = np.dot(np.linalg.inv(Tw_g), Tw_n)
	# print(Tg_n)
	# T1 = np.dot(np.linalg.inv(Tg_n), np.array([0.15090656, -0.40494069, 1]).reshape(3, 1))
	# print(T1)
	
	Xr = []; Yr = []; THETAr = []
	for i in range(len(X1)):
		p = (X1[i], Y1[i], THETA1[i])
		Tn_i = np.array([[math.cos(p[2]), -math.sin(p[2]), p[0]], [math.sin(p[2]), math.cos(p[2]), p[1]], [0, 0, 1]])
		Tg_i = np.dot(Tg_n, Tn_i)
		x = Tg_i[0][2]
		y = Tg_i[1][2]
		theta = math.atan2(Tg_i[1, 0], Tg_i[0, 0])
		Xr.append(x); Yr.append(y); THETAr.append(theta)

	# for i in range(len(X1)):
	# 	p = np.array([X1[i], Y1[i], 1]).reshape(3, 1)
	# 	Tg_i = np.dot(Tg_n, p)
	# 	Xr.append(Tg_i[0, 0]); Yr.append(Tg_i[1, 0])

	return (Xr, Yr, THETAr)



if __name__ == '__main__':
	fileNoise = str(argv[1])
	fileGt = str(argv[2])

	(X1, Y1, THETA1, LBL) = readTxt(fileNoise)
	(X2, Y2, THETA2, LBL) = readTxt(fileGt)
	
	X1 = X1[0:2800]; Y1 = Y1[0:2800]
	X2 = X2[0:2800]; Y2 = Y2[0:2800]

	# draw(X1, Y1, X2, Y2)

	(Xr, Yr, THETAr) = rectify(X1, Y1, THETA1, X2, Y2, THETA2)
	# print((Xr[0], Yr[0]), (X2[0], Y2[0]))

	# draw(Xr, Yr, X2, Y2)