# Usage : python mlp_read.py mlp_in.txt
# Output : Displays current node number on terminal


from sys import argv, exit
import matplotlib.pyplot as plt
import numpy as np


def read(fileName):
	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	Node_meta = []

	for line in A:
		(l1, b1, l2, b2, lbl) = line.split(' ')
		Node_meta.append((float(l1), float(b1), float(l2), float(b2), float(lbl.rstrip('\n'))))

	return Node_meta


fig = plt.figure()
ax = plt.subplot(111)

def on_plot_hover(event):
	for line in ax.get_lines():
		if line.contains(event)[0]:
			print("Over %s node" % line.get_gid())


def drawNode(Node_meta):
	for i, line in enumerate(Node_meta):
		lbl = line[4]
		x = [line[0], line[2]]
		y = [line[1], line[3]]

		if lbl == 0:
			ax.plot(x, y, 'ro')
			ax.plot(x, y, 'r-', gid=i+1)

		elif lbl == 1:
			ax.plot(x, y, 'bo')
			ax.plot(x, y, 'b-', gid=i+1)

		elif lbl == 2:
			ax.plot(x, y, 'go')
			ax.plot(x, y, 'g-', gid=i+1)

		elif lbl == 3:
			ax.plot(x, y, 'yo')
			ax.plot(x, y, 'y-', gid=i+1)

	fig.canvas.mpl_connect('motion_notify_event', on_plot_hover)

	plt.show()


def printNodes(Node_meta):
	np.set_printoptions(precision=6)
	np.set_printoptions(suppress=True)
	Node_meta = np.array(Node_meta)
	print repr(Node_meta)


if __name__ == '__main__':
	fileName = str(argv[1])
	Node_meta = read(fileName)

	drawNode(Node_meta)

	printNodes(Node_meta)