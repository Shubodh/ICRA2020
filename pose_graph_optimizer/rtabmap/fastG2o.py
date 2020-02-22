from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import csv

from manh_const import startPoses, draw


# def readCsv(fileName):
# 	pairs = []
# 	with open(fileName, 'rt') as f:
# 		A = csv.reader(f)

# 		for line in A:
# 			pair = []
			
# 			pair.append(float(line[0]))
			
# 			if line[1] == 's': pair.append(0)
# 			else: pair.append(1)
			
# 			pair.append(float(line[2]))
			
# 			if line[3] == 's': pair.append(0)
# 			else: pair.append(1)

# 			pairs.append(pair)

# 	return pairs


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


def readLabels(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	for i, lbl in enumerate(A):
		lbl = lbl.rstrip('\n')
		if(lbl == 'Rackspace'):
			A[i] = 0
		elif(lbl == 'Corridor'):
			A[i] = 1
		elif(lbl == 'Transition'):
			A[i] = 2

	return A


def readMLPOut(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	mlpN = []
	for line in A:
		l = line.split(' ')
		mlpN.append((int(l[0]), int(l[1].rstrip('\n'))))

	return mlpN


def writeG2O(X, Y, THETA, poses, mlpN):
	# g2o = open('/run/user/1000/gvfs/sftp:host=10.2.138.226,user=udit/home/udit/backup/lessNoise.g2o', 'w')
	g2o = open('lessNoise.g2o', 'w')

	for i, (x, y, theta) in enumerate(zip(X, Y, THETA)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	# Odometry
	# T1_w : 1 with respect to world
	g2o.write("# Odometry constraints\n")
	info_mat = "500.0 0.0 0.0 500.0 0.0 500.0"
	for i in range(1, len(X)):
		p1 = (X[i-1], Y[i-1], THETA[i-1])
		p2 = (X[i], Y[i], THETA[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(i-1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
		g2o.write(line)

	# Removing rotation of 1st corridor
	g2o.write("\n# Removing rotation of 1st corridor\n\n")
	for i in range(1, 20):
		p1 = (poses[0, 0], poses[0, 1], poses[0, 2])
		p2 = (poses[i, 0], poses[i, 1], poses[i, 2])
		startId = int(poses[0, 3]); denseId = int(poses[i, 3])

		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(startId)+" "+str(denseId)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# MLP Manhattan constraints
	g2o.write("\n# MLP Manhattan constraints\n\n")
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
			startId = int(poses[e1, 3]); endId = int(poses[e2, 3])

			T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
			T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
			T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
			del_x = str(T2_1[0][2])
			del_y = str(T2_1[1][2])
			del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
			
			line = "EDGE_SE2 "+str(startId)+" "+str(endId)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
			g2o.write(line)

	g2o.write("FIX 0\n")
	g2o.close()	


def optimize():
	cmd = "g2o -robustKernel Cauchy -robustKernelWidth 1 -o opt.g2o -i 50 lessNoise.g2o > /dev/null 2>&1"
	os.system(cmd)


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

	return (X, Y, THETA)


if __name__ == '__main__':
	(X, Y, THETA) = readKitti(argv[1])
	lbls = readLabels(argv[2])

	poses = startPoses(X, Y, THETA, lbls)
	poses = np.asarray(poses)

	mlpN = readMLPOut(argv[3])

	# frtLoop = readCsv(argv[4])

	writeG2O(X, Y, THETA, poses, mlpN)

	optimize()
	(xOpt, yOpt, tOpt) = readG2o("opt.g2o")
	draw(xOpt, yOpt, lbls)
