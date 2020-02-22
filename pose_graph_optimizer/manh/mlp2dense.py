# Usage : python mlp2dense.py mlp_output.csv mlp_in_dense.txt
# Output : mlp_out_dense.txt


from sys import argv
import matplotlib.pyplot as plt
import math
import numpy as np
import csv


def readMlp(fileName):
	nodes1 = []
	nodes2 = []

	with open(fileName, 'rt') as f:
		A = csv.reader(f)

		for n1, n2 in A:
			# Mlp's 1 based indexing
			nodes1.append(int(n1)-1)
			nodes2.append(int(n2)-1)

	return (nodes1, nodes2)


def readPoses(fileName):
	poses = []

	f = open(fileName, 'r')
	A = f.readlines()
	f.close()

	for line in A:
		(st, end) = line.split(' ')
		poses.append((int(st), int(end.rstrip('\n'))))

	return poses


def writeDense(nodes1, nodes2, poses):
	outFile = open("mlp_out_dense.txt", 'w')

	for i in range(len(nodes1)):
		n1 = nodes1[i]; n2 = nodes2[i]
		# print(poses[n1], poses[n2])
		info = str(poses[n1][0])+" "+str(poses[n1][1])+" "+str(poses[n2][0])+" "+str(poses[n2][1])
		outFile.write(info)
		outFile.write("\n")

	outFile.close()


if __name__ == '__main__':
	fileName1 = str(argv[1])
	(nodes1, nodes2) = readMlp(fileName1)

	fileName2 = str(argv[2])
	poses = readPoses(fileName2)

	writeDense(nodes1, nodes2, poses)