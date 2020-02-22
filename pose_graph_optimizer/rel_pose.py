import math
import numpy as np

if __name__ == '__main__':
	p1 = (2, 3, math.pi/6)
	p2 = (5, 7, math.pi/3)

	dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**(0.5)
	delTheta = p2[2] - p1[2]

	delX = dist * math.cos(delTheta)
	delY = dist * math.sin(delTheta)

	print(delX, delY, np.degrees(delTheta))

	# homo = np.array([[math.cos(delTheta), -math.sin(delTheta), 0, 2], [math.sin(delTheta), math.cos(delTheta), 0, 3], [0,0,1,0], [0,0,0,1]])
	# relVec = np.array([delX, delY, 0, 1]).reshape(4,1)
	# print(np.dot(homo, relVec))

	# vec2 = np.array([5,7,0,1])
	# rel2_1 = np.dot(np.linalg.inv(homo), vec2)
	# print(rel2_1)

	T1_w = np.array([[math.cos(p1[2]), -math.sin(p1[2]), p1[0]], [math.sin(p1[2]), math.cos(p1[2]), p1[1]], [0, 0, 1]])
	T2_w = np.array([[math.cos(p2[2]), -math.sin(p2[2]), p2[0]], [math.sin(p2[2]), math.cos(p2[2]), p2[1]], [0, 0, 1]])
	T2_1 = np.dot(np.linalg.inv(T1_w), T2_w)
	print(T2_1)
	print(T2_1[0][0], np.arccos(T2_1[0][0])*(180/math.pi))