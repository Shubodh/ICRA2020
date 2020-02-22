import matplotlib.pyplot as plt
from sys import argv

def read(fileName):
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
	fileName = str(argv[1])
	(X, Y, THETA) =	read(fileName)

	plt.plot(X, Y, 'bo')
	plt.plot(X, Y, 'k')
	# plt.xlim(-20, 5)
	# plt.ylim(-20, 5)
	plt.show()	
