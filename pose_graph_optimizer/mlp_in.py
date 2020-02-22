from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np


def getTheta(X ,Y):
	THETA = [None]*len(X)
	for i in xrange(1, len(X)-1):
		if(X[i+1] == X[i-1]):
			if (Y[i+1]>Y[i-1]):
				THETA[i] = math.pi/2
			else:
				THETA[i] = 3*math.pi/2
			continue

		THETA[i] = math.atan((Y[i+1]-Y[i-1])/(X[i+1]-X[i-1]))

		if(X[i+1]-X[i-1] < 0):
			THETA[i] += math.pi 

	if X[1]==X[0]:
		if Y[1] > Y[0]:
			THETA[0] = math.pi/2
		else:
			THETA[0] = 3*math.pi/2
	else:
		THETA[0] = math.atan((Y[1]-Y[0])/(X[1]-X[0]))

	if X[-1] == X[len(Y)-2]:
		if Y[1] > Y[0]:
			THETA[-1] = math.pi/2
		else:
			THETA[-1] = 3*math.pi/2
	else:
		THETA[-1] = math.atan((Y[-1]-Y[len(Y)-2])/(X[-1]-X[len(Y)-2]))

	return THETA


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
		LBL.append(int(lbl.rstrip('\n')))

	X_temp = X
	Y_temp = Y
	X = [-y for y in Y_temp]
	Y = [x for x in X_temp]

	THETA = getTheta(X, Y)

	return (X, Y, THETA, LBL)

def meta(X, Y, THETA, LBL):
	Node_meta = []
	Node_mid = []
	st = end = 0

	for i in xrange(1, len(LBL)):
		if LBL[i] == LBL[i-1]:
			end = i
			continue

		mid = st + (end - st)/2
		
		Node_meta.append((X[st], Y[st], X[end], Y[end], LBL[mid]))
		Node_mid.append((X[mid], Y[mid], THETA[mid]))

		st = end + 1
		end = st
	return (Node_meta, Node_mid)


def drawNode(Node_meta, Node_mid):
	ax = plt.subplot(1,1,1)

	X = []; Y = []

	for line in Node_meta:
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		if lbl == 0:
			ax.plot(x, y, 'ro')
			ax.plot(x, y, 'r-')

		elif lbl == 1:
			ax.plot(x, y, 'bo')
			ax.plot(x, y, 'b-')

		elif lbl == 2:
			ax.plot(x, y, 'go')
			ax.plot(x, y, 'g-')

		elif lbl == 3:
			ax.plot(x, y, 'yo')
			ax.plot(x, y, 'y-')

	X_mid = []; Y_mid = []; THETA_mid = []
	for e in Node_mid:
		X_mid.append(e[0]); Y_mid.append(e[1]); THETA_mid.append(e[2])

	# plt.plot(X_mid, Y_mid, 'mo')

	# for i in xrange(len(X_mid)):
	# 	x2 = math.cos(THETA_mid[i]) + X_mid[i]
	# 	y2 = math.sin(THETA_mid[i]) + Y_mid[i]
	# 	plt.plot([X_mid[i], x2], [Y_mid[i], y2], 'm->')

	plt.xlim(-5, 25)
	plt.ylim(-15, 15)
	plt.show()

if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA, LBL) = read(fileName)

	(Node_meta, Node_mid) = meta(X, Y, THETA, LBL)

	# Node_meta = Node_meta[7:-1]; Node_mid = Node_mid[7: -1]

	drawNode(Node_meta, Node_mid)
	print(np.array(Node_meta))
	# poses = open("mlp_in.txt", 'w')
	# for line in Node_meta:
	# 	info = str(line[0])+" "+str(line[1])+" "+ str(line[2])+" "+ str(line[3])+" "+ str(line[4]) 
	# 	poses.write(info)
	# 	poses.write("\n")

	# poses.close()
	