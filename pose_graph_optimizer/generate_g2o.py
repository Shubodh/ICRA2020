from sys import argv
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
	fileName = str(argv[1])
	(X, Y, THETA, LBL) = read(fileName)
	
	X_meta = []
	Y_meta = []
	THETA_meta = []
	st = end = 0

	for i in xrange(1, len(LBL)):
		if LBL[i] == LBL[i-1]:
			end = i
			continue

		mid = st + (end - st)/2
		X_meta.append(X[mid])
		Y_meta.append(Y[mid])
		THETA_meta.append(THETA[mid])

		st = end + 1
		end = st

	g2o = open("poses.g2o", 'w')
	for i, (x, y, theta) in enumerate(zip(X_meta,Y_meta,THETA_meta)):
		line = "VERTEX_SE2 " + str(i) + " " + str(x) + " " + str(y) + " " + str(theta)
		g2o.write(line)
		g2o.write("\n")

	info_mat = "500.0 0.0 0.0 500.0 0.0 500.0"
	for i in xrange(1, len(X_meta)):
		del_x = str(X_meta[i] - X_meta[i-1])
		del_y = str(Y_meta[i] - Y_meta[i-1])
		del_theta = str(THETA_meta[i] - THETA_meta[i-1])
		line = "EDGE_SE2 "+str(i-1)+" "+str(i)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat
		g2o.write(line)
		g2o.write("\n")


	g2o.write("FIX 0")
	g2o.write("\n")

	# Loop closure constraints
	
	info_mat = "1000.0 0.0 0.0 1000.0 0.0 1000.0"
	del_x = str(0.02)
	v1 = 8; v2 = 12; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge1 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	v1 = 9; v2 = 11; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge2 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	# v1 = 17; v2 = 31; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	# edge3 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	v1 = 18; v2 = 30; del_y = str(1.2); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge4 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+str(-1)+" "+str(1.2)+" "+del_theta+" "+info_mat+"\n"
	# v1 = 19; v2 = 29; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	# edge5 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	v1 = 22; v2 = 26; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge6 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+str(-2)+" "+str(1)+" "+del_theta+" "+info_mat+"\n"
	v1 = 32; v2 = 58; del_y = str(0.02); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge7 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	# v1 = 33; v2 = 57; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	# edge8 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	# v1 = 39; v2 = 48; del_y = str(Y_meta[v2] - Y_meta[v1]); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	# edge9 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	v1 = 3; v2 = 15; del_y = str(0); del_theta = str(THETA_meta[v2] - THETA_meta[v1])
	edge10 = "EDGE_SE2 "+str(v1)+" "+str(v2)+" "+del_x+" "+del_y+" "+del_theta+" "+info_mat+"\n"
	
	g2o.write("\n")
	g2o.write(edge1)
	g2o.write(edge2)
	# g2o.write(edge3)
	g2o.write(edge4)
	# g2o.write(edge5)
	g2o.write(edge6)
	g2o.write(edge7)
	# g2o.write(edge8)
	# g2o.write(edge9)
	g2o.write(edge10)

	g2o.close()

	plt.plot(X_meta, Y_meta, 'ro')
	plt.plot(X, Y, 'k')
	plt.plot(X_meta[15], Y_meta[15], 'bo')
	plt.plot(X_meta[3], Y_meta[3], 'bo')
	plt.xlim(-15,15)
	plt.ylim(-25,5)
	plt.show()

