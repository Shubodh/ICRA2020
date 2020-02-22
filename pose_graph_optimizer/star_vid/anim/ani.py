import numpy as np
import matplotlib.pyplot as plt


def draw(X, Y):
	plt.plot(X, Y, 'ro')
	plt.xlim(0, 2*np.pi)
	plt.ylim(-1, 1)
	plt.show(block=False)
	plt.pause(0.001)
	# plt.close()


if __name__ == '__main__':
	X = np.linspace(0, 2*np.pi, 2000)
	Y = np.sin(X)

	for i in range(0, len(X), 20):
		draw(X[0:i], Y[0:i])