# Usage : python fast/fastG2o2.py unoptimised_tracks.csv
# Output : lessNoise.g2o file in current directory

from sys import argv, exit
import matplotlib.pyplot as plt
import math
import numpy as np
import csv


def getTheta(X ,Y):
	THETA = [None]*len(X)
	for i in range(1, len(X)-1):
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
				# THETA.append(float(line[3]))
				LBL.append(float(line[4]))

	X_temp = X
	Y_temp = Y
	X = [-y for y in Y_temp]
	Y = [x for x in X_temp]

	THETA = getTheta(X, Y)

	return (X, Y, THETA, LBL)


# def calcTheta(x1, x2, y1, y2):
# 	if(x2 == x1):
# 		if(y2 > y1):
# 			theta = math.pi/2
# 		else:
# 			theta = 3*math.pi/2
# 	else:
# 		theta = math.atan((y2-y1)/(x2-x1))

# 	if(x2-x1 < 0):
# 		theta += math.pi

# 	return theta	


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

fig = plt.figure()
ax = plt.subplot(111)

def on_plot_hover(event):
	for pt in ax.get_lines():
		if pt.contains(event)[0]:
			print("over %s" % pt.get_gid())
			

def draw(X, Y, LBL):
	for i in range(len(LBL)):
		x = X[i]; y = Y[i]
		if LBL[i] == 0:
			ax.plot(x, y, 'ro', gid=i, zorder = 2, markersize=5)

		elif LBL[i] == 1:
			ax.plot(x, y, 'bo', gid=i, zorder = 4, markersize=5)

		elif LBL[i] == 2:
			ax.plot(x, y, 'go', gid=i, zorder = 6, markersize=5)

		elif LBL[i] == 3:
			ax.plot(x, y, 'yo', gid=i, zorder = 8, markersize=5)


	plt.plot(X, Y, 'k-')

	fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

	plt.show()


def writeG2O(X_meta,Y_meta,THETA_meta):
	# sz = int(len(X_meta))
	# X_meta = X_meta[0:sz]; Y_meta = Y_meta[0:sz]; THETA_meta = THETA_meta[0:sz]

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

	# Section I
	g2o.write("# Section I constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 400.0 0.0 1000.0"
	i = 212; ii = 6848
	del_x = str(0.1); del_y = str(0.5); del_theta = str(0)
	
	for cnt in range(38):
		line = "EDGE_SE2 "+str(i)+" "+str(ii)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")	
		i += 1; ii += 1	

	# Section II
	g2o.write("# Section II constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 1000.0"
	del_x = str(0.1); del_y = str(0.3); del_theta = str(0)
	edges = [(5952, 262), (5956, 265), (5960, 488), (5964, 493), (5968, 758), (5972, 765), (5976, 1001), \
			(5980, 1006), (5984, 1223), (5988, 1229), (5992, 1449), (5996, 1454), (6000, 1675), (6004, 1680), \
			(6008, 1905), (6012, 1909)]

	for e in edges:
		line = "EDGE_SE2 "+str(e[1])+" "+str(e[0])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Section III
	g2o.write("# Section III constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 1000.0"
	del_x = str(0.1); del_y = str(0.1); del_theta = str(1.5*math.pi)
	edges = [(142, 6456), (370, 6465), (619, 6474), (887, 6483), (1110, 6492), (1333, 6501), (1560, 6510), \
			(1787, 6519), (2017, 6528), (2245, 6537), (2495, 6546), (2750, 6555), (3047, 6564), (3304, 6573), \
			(3562, 6582), (3822, 6591), (4082, 6600), (4342, 6609), (4600, 6617)]

	for e in edges:
		line = "EDGE_SE2 "+str(e[0])+" "+str(e[1])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Section IV
	g2o.write("# Section IV constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 700.0"
	del_x = str(0.1); del_y = str(0.1); del_theta = str(0)
	edges = [(2623, 6044), (2627, 6049), (2900, 6054), (2905, 6058), (3180, 6063), (3185, 6068), (3437, 6073),\
			(3443, 6078), (3697, 6082), (3702, 6087), (3955, 6092), (3960, 6097), (4214, 6102),\
			(4220, 6107), (4476, 6111), (4480, 6116), (4731, 6121), (4735, 6126)]

	for e in edges:
		line = "EDGE_SE2 "+str(e[0])+" "+str(e[1])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Section V
	g2o.write("# Section V constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 1000.0"
	del_x = str(0.1); del_y = str(0.1); del_theta = str(math.pi)
	edges = [(2144, 2358), (2172, 2330)]

	for e in edges:
		line = "EDGE_SE2 "+str(e[0])+" "+str(e[1])+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Section VI
	g2o.write("# Section VI constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 700.0"

	for i in range(1, 235):
		p1 = (X_meta[1], Y_meta[1], THETA_meta[1])
		p2 = (X_meta[i], Y_meta[i], THETA_meta[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	# Section VII
	g2o.write("# Section VII constraints")
	g2o.write("\n")
	info_mat = "700.0 0.0 0.0 700.0 0.0 700.0"

	for i in range(6393, 6505):
		p1 = (X_meta[6392], Y_meta[6392], THETA_meta[6392])
		p2 = (X_meta[i], Y_meta[i], THETA_meta[i])
		T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
		T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
		T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
		del_x = str(T2_1[0][2])
		del_y = str(T2_1[1][2])
		del_theta = str(math.atan2(T2_1[1, 0], T2_1[0, 0]))
		
		line = "EDGE_SE2 "+str(6392)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")

	g2o.write("FIX 0")
	g2o.write("\n")
	g2o.close()


if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA, LBL) = read(fileName)
	print(len(X))
	# draw(X, Y, LBL)

	# drawTheta(X, Y, LBL, THETA)

	writeG2O(X, Y, THETA)