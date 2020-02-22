from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np
import csv

import manhCsv3 as m3
import manhCsv4 as m4


def readG2o(fileG2o, fileTxt):
	f = open(fileG2o, 'r')
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

	f = open(fileTxt, 'r')
	A = f.readlines()
	f.close()

	LBL = []

	for line in A:
		(x, y, theta, lbl) = line.split(' ')
		LBL.append(int(lbl.rstrip('\n')))

	X_temp = X
	Y_temp = Y
	X = [y for y in Y_temp]
	Y = [-x for x in X_temp]

	# THETA = getTheta(X, Y)

	return (X, Y, THETA, LBL)


def readCsv(fileName):
	X = []
	Y = []
	THETA = []
	LBL = []

	with open(fileName, 'rt') as f:
		A = csv.reader(f)

		for idx, line in enumerate(A):
			if(idx == 0):
				continue
			else:
				X.append(float(line[1]))
				Y.append(float(line[2]))
				THETA.append(float(line[3]))
				LBL.append(float(line[4]))

	return (X, Y, THETA, LBL)


def readMlp(fileName):
	mlpN = []

	with open(fileName, 'rt') as f:
		A = csv.reader(f)

		for line in A:
			mlpN.append((int(line[0]), int(line[1])))

	return mlpN


def anim(Nodes):
	fig = plt.figure("Animation")

	for i, line in enumerate(Nodes):
		l1=line[0]; b1=line[1]; l2=line[2]; b2=line[3]; lbl=line[4]; stPose=line[5]; endPose=line[6]

		x = [l1, l2]
		y = [b1, b2]

		if lbl == 0:
			plt.plot(x, y, 'ro', markersize=5)
			plt.plot(x, y, 'r-', linewidth=2)

		elif lbl == 1:
			plt.plot(x, y, 'bo', markersize=5)
			plt.plot(x, y, 'b-', linewidth=2)

		elif lbl == 2:
			plt.plot(x, y, 'go', markersize=5)
			plt.plot(x, y, 'g-', linewidth=2)

		elif lbl == 3:
			plt.plot(x, y, 'yo', markersize=5)
			plt.plot(x, y, 'y-', linewidth=2)

	for line in mlpN:
		t = line[0]-1; s = line[1]-1
		if(len(Nodes)-1 == t):
			line1 = Nodes[s]; line2 = Nodes[t]
			
			l1=line1[0]; b1=line1[1]; l2=line1[2]; b2=line1[3]; lbl=line1[4]			
			x = [l1, l2]; y = [b1, b2]

			if lbl == 0:
				plt.plot(x, y, 'ro', markersize=3.5)
				plt.plot(x, y, 'r-', linewidth=4)

			elif lbl == 1:
				plt.plot(x, y, 'bo', markersize=3.5)
				plt.plot(x, y, 'b-', linewidth=4)

			elif lbl == 2:
				plt.plot(x, y, 'go', markersize=3.5)
				plt.plot(x, y, 'g-', linewidth=4)

			elif lbl == 3:
				plt.plot(x, y, 'yo', markersize=3.5)
				plt.plot(x, y, 'y-', linewidth=4)


			l1=line2[0]; b1=line2[1]; l2=line2[2]; b2=line2[3]; lbl=line2[4]			
			x = [l1, l2]; y = [b1, b2]

			if lbl == 0:
				plt.plot(x, y, 'ro', markersize=3.5)
				plt.plot(x, y, 'r-', linewidth=4)

			elif lbl == 1:
				plt.plot(x, y, 'bo', markersize=3.5)
				plt.plot(x, y, 'b-', linewidth=4)

			elif lbl == 2:
				plt.plot(x, y, 'go', markersize=3.5)
				plt.plot(x, y, 'g-', linewidth=4)

			elif lbl == 3:
				plt.plot(x, y, 'yo', markersize=3.5)
				plt.plot(x, y, 'y-', linewidth=4)
			
			break			

	plt.xlim(-50, 10)
	plt.ylim(-25, 15)

	if(len(Nodes) <= 325):
		plt.show(block=False)
	else:
		plt.show(block=False)
	
	plt.pause(0.1)
	plt.clf()


if __name__ == '__main__':
	# fileG2o = str(argv[1]); fileTxt = str(argv[2]); fileMlp = str(argv[3])
	# (X, Y, THETA, LBL) = readG2o(fileG2o, fileTxt)
	# Nodes = m3.start(X, Y, THETA, LBL)
	# mlpN = readMlp(fileMlp)

	fileNoise = str(argv[1]); fileMlp = str(argv[2])
	(X, Y, THETA, LBL) = readCsv(fileNoise)
	Nodes = m4.start(X, Y, THETA, LBL)
	mlpN = readMlp(fileMlp)

	for i in range(len(Nodes)):
		anim(Nodes[0:i])
