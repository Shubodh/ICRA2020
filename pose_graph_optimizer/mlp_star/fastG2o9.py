from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np
import csv

import manh_constraint9 as const
from icpDense import icpPoses as iPoses


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
		LBL.append(int(lbl.rstrip('\n')))

	return (X, Y, THETA, LBL)


def readCsv(fileName):
	mlpN = []

	with open(fileName, 'rt') as f:
		A = csv.reader(f)

		for line in A:
			mlpN.append((int(line[0]), int(line[1])))

	return mlpN


def drawTheta(X, Y, LBL, thetas):
	ax = plt.subplot(111)

	X0 = []; Y0 = []; X1 = []; Y1 = []; X2 = []; Y2 =[]; X3 = []; Y3 = [];
	
	for i in range(len(LBL)):

		x2 = math.cos(thetas[i]) + X[i]
		y2 = math.sin(thetas[i]) + Y[i]
		plt.plot([X[i], x2], [Y[i], y2], 'm->')

		if LBL[i] == 0:
			X0.append(X[i])
			Y0.append(Y[i])

		elif LBL[i] == 1:
			X1.append(X[i])
			Y1.append(Y[i])

		elif LBL[i] == 2:
			X2.append(X[i])
			Y2.append(Y[i])

		elif LBL[i] == 3:
			X3.append(X[i])
			Y3.append(Y[i])

	ax.plot(X0, Y0, 'ro', label='Rackspace', zorder = 2)
	ax.plot(X1, Y1, 'bo', label='Corridor', zorder = 4)
	ax.plot(X2, Y2, 'go', label='Trisection', zorder = 6)
	ax.plot(X3, Y3, 'yo', label='Intersection', zorder = 8)

	plt.plot(X, Y, 'k-')

	plt.show()


def draw(X, Y, LBL):
	X0 = []; Y0 = []; X1 = []; Y1 = []; X2 = []; Y2 =[]; X3 = []; Y3 = [];
	
	for i in xrange(len(LBL)):
		if LBL[i] == 0:
			X0.append(X[i])
			Y0.append(Y[i])

		elif LBL[i] == 1:
			X1.append(X[i])
			Y1.append(Y[i])

		elif LBL[i] == 2:
			X2.append(X[i])
			Y2.append(Y[i])

		elif LBL[i] == 3:
			X3.append(X[i])
			Y3.append(Y[i])

	fig = plt.figure()
	ax = plt.subplot(111)

	ax.plot(X0, Y0, 'ro', label='Rackspace')
	ax.plot(X1, Y1, 'bo', label='Corridor')
	ax.plot(X2, Y2, 'go', label='Trisection')
	ax.plot(X3, Y3, 'yo', label='Intersection')
	plt.plot(X, Y, 'k-')

	plt.show()


def writeG2O(X_meta, Y_meta, THETA_meta, poses, mlpN, iPoses, icpId):
	g2o = open('/run/user/1000/gvfs/sftp:host=ada.iiit.ac.in,user=udit/home/udit/share/lessNoise.g2o', 'w')
	
	for i, (x, y, theta) in enumerate(zip(X_meta,Y_meta,THETA_meta)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	# Odometry
	g2o.write("# Odometry constraints")
	g2o.write("\n")
	info_mat = "500.0 0.0 0.0 500.0 0.0 500.0"
	for i in range(1, len(X_meta)):
		p1 = (X_meta[i-1], Y_meta[i-1], THETA_meta[i-1])
		p2 = (X_meta[i], Y_meta[i], THETA_meta[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(i-1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Manhattan constraints
	# g2o.write("# Manhattan constraints")
	# g2o.write("\n")
	# info_mat = "300.0 0.0 0.0 300.0 0.0 700.0"

	# for i in range(1, len(poses)):
	# 	p1 = (poses[0, 0], poses[0, 1], poses[0, 2])
	# 	p2 = (poses[i, 0], poses[i, 1], poses[i, 2])
	# 	startId = int(poses[0, 3]); denseId = int(poses[i, 3])

	# 	T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
	# 	T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
	# 	T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
	# 	del_x = str(T2_1[0][2])
	# 	del_y = str(T2_1[1][2])
	# 	del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
	# 	line = "EDGE_SE2 "+str(startId)+" "+str(denseId)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
	# 	g2o.write(line)
	# 	g2o.write("\n")

	# MLP Manhattan constraints
	g2o.write("\n# MLP Manhattan constraints\n\n")
	# g2o.write("\n")
	info_mat = "300.0 0.0 0.0 300.0 0.0 700.0"
	for i in range(len(mlpN)):
		n1 = mlpN[i][1]; n2 = mlpN[i][0]
		s1 = 2*n1; s2 = s1+1; t1 = 2*n2; t2 = t1+1  
		
		# s1 -> t1; s1 -> t2; s2 -> t1; s2 -> t2
		pairs = [(s1, t1), (s1, t2), (s2, t1), (s2, t2)]
		
		for j in range(len(pairs)):
			e1 = pairs[j][0]; e2 = pairs[j][1]

			p1 = (poses[e1, 0], poses[e1, 1], poses[e1, 2])
			p2 = (poses[e2, 0], poses[e2, 1], poses[e2, 2])
			startId = int(poses[e1, 3]); denseId = int(poses[e2, 3])

			T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
			T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
			T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
			del_x = str(T2_1[0][2])
			del_y = str(T2_1[1][2])
			del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
			
			line = "EDGE_SE2 "+str(startId)+" "+str(denseId)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
			g2o.write(line)
			# g2o.write("\n")

	# ICP constraints
	g2o.write("# ICP constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 900.0"

	for i in range(icpId.shape[0]):
		x = iPoses[i, 0]; y = iPoses[i, 1]; theta = iPoses[i, 2]; frame1 = icpId[i, 0]; frame2 = icpId[i, 1]

		if(frame1 == frame2):
			continue

		line = "EDGE_SE2 "+str(frame1)+" "+str(frame2)+" "+str(x)+" "+str(y)+" "+str(theta)+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	g2o.write("FIX 0\n")
	# g2o.write("\n")
	g2o.close()


if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA, LBL) = read(fileName)
	X = X[0:2800]; Y = Y[0:2800]; LBL = LBL[0:2800]
	# X = X[50:2800]; Y = Y[50:2800]; LBL = LBL[50:2800]

	# draw(X, Y, LBL)
	# drawTheta(X, Y, LBL, THETA)

	poses, icpId = const.start(fileName)
	poses = np.asarray(poses); icpId = np.asarray(icpId)
	# print(poses.shape, poses[0, :])

	# 60 pairs from MLP = 143 nodes in MG of star1
	# mlpN = readCsv(str(argv[2]))[0:60][:]
	mlpN = readCsv(str(argv[2]))
	# print(len(mlpN))

	writeG2O(X, Y, THETA, poses, mlpN, iPoses, icpId)
