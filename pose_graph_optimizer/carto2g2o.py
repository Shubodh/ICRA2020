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
		LBL.append(float(lbl.rstrip('\n')))

	return (X, Y, THETA, LBL)


def meta(X, Y, THETA, LBL):
	X_meta = []
	Y_meta = []
	THETA_meta = []
	LBL_meta = []
	st = end = 0

	for i in xrange(1, len(LBL)):
		if LBL[i] == LBL[i-1]:
			end = i
			continue

		mid = st + (end - st)/2
		X_meta.append(X[mid])
		Y_meta.append(Y[mid])
		THETA_meta.append(THETA[mid])
		LBL_meta.append(LBL[mid])

		st = end + 1
		end = st
	return (X_meta, Y_meta, THETA_meta, LBL_meta)


def writeG2O(X_meta,Y_meta,THETA_meta):
	g2o = open('/run/user/1000/gvfs/sftp:host=ada.iiit.ac.in,user=udit/home/udit/share/poses.g2o', 'w')
	for i, (x, y, theta) in enumerate(zip(X_meta,Y_meta,THETA_meta)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	# info_mat = "1000.0 0.0 0.0 1000.0 0.0 0.00001"
	info_mat = "500.0 0.0 0.0 500.0 0.0 500.0"
	for i in xrange(1, len(X_meta)):
		p1 = (X_meta[i-1], Y_meta[i-1], THETA_meta[i-1])
		p2 = (X_meta[i], Y_meta[i], THETA_meta[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		# Devil below
		# del_theta = str(np.arccos(T2_1[0][0]))
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(i-1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")


	e1 = (12, 20); e2 = (16, 18); e3 = (15, 19)
	e4 = (31, 35); e5 = (25, 39); e6 = (25, 39); 
	e7 = (49, 57); e8 = (46, 60);
	edges = [e1, e2, e3, e4, e5, e6, e7, e8]

	info_mat = "200.0 0.0 0.0 1000.0 0.0 1000.0"
	for e in edges:
		p1 = (X_meta[e[0]], Y_meta[e[0]], THETA_meta[e[0]])
		p2 = (X_meta[e[1]], Y_meta[e[1]], THETA_meta[e[1]])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(0)
		del_theta = str(math.pi)
		line = "EDGE_SE2 "+str(e[0])+" "+str(e[1])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	corWidth = 1.789
	edges = [(12, 26), (16, 32), (14, 28), (27, 47), (32, 50)]

	info_mat = "200.0 0.0 0.0 1000.0 0.0 1000.0"
	for e in edges:
		p1 = (X_meta[e[0]], Y_meta[e[0]], THETA_meta[e[0]])
		p2 = (X_meta[e[1]], Y_meta[e[1]], THETA_meta[e[1]])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(-corWidth)
		del_theta = str(0)
		line = "EDGE_SE2 "+str(e[0])+" "+str(e[1])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	g2o.write("FIX 0")
	g2o.write("\n")
	g2o.close()	


def draw_meta(X, Y, LBL):
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

	ax = plt.subplot(1,1,1)
	ax.plot(X0, Y0, 'ro', label='Rackspace')
	ax.plot(X1, Y1, 'bo', label='Corridor')
	ax.plot(X2, Y2, 'go', label='Trisection')
	ax.plot(X3, Y3, 'yo', label='Intersection')

	plt.plot(X_meta, Y_meta, 'k')

	plt.show(block=True)


def draw(X_meta, Y_meta, THETA_meta, theta=True):
	plt.plot(X_meta, Y_meta, 'bo')
	plt.plot(X_meta, Y_meta, 'k')

	if (theta == True):
		for i in xrange(len(X_meta)):
			x2 = math.cos(THETA_meta[i]) + X_meta[i]
			y2 = math.sin(THETA_meta[i]) + Y_meta[i]
			plt.plot([X_meta[i], x2], [Y_meta[i], y2], 'r->')
	
	plt.plot(X_meta[32], Y_meta[32], 'ro')
	plt.plot(X_meta[50], Y_meta[50], 'ro')
	# plt.plot(X_meta[39], Y_meta[39], 'ro')
	# plt.plot(X_meta[48], Y_meta[48], 'ro')
	plt.show()


if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA, LBL) = read(fileName)
	sz = len(X)
	beg = 0
	X = X[beg:sz/4]; Y = Y[beg:sz/4]; THETA = THETA[beg:sz/4]; LBL = LBL[beg:sz/4]

	sz = len(X)
 	ed = 780
	X = X[0:ed]; Y = Y[0:ed]; THETA = THETA[0:ed]; LBL = LBL[0:ed]

	(X_meta, Y_meta, THETA_meta, LBL_meta) = meta(X, Y, THETA, LBL)

	writeG2O(X_meta,Y_meta,THETA_meta)
	
	draw_meta(X_meta, Y_meta, LBL_meta)
	draw(X_meta, Y_meta, THETA_meta, theta=False)

