import torch
#import torchvision
import torch.nn
import numpy as np
import random
import torch.nn.functional as F
import sys
#import matplotlib.pyplot as plt


device = 'cuda'

#hyper parameters
input_size = 4
output_size = 8
hl1_size = 70
hl2_size = 90
hl3_size = 15
hl4_size = 16
num_epochs = 75
learning_rate = 0.001

import torch
import torch.nn


x_test = np.array([[0,0,8.60269753278,0.0,0],
[8.60269753278,0.0,10.1607962984,0.0,2],
[10.1607962984,0.0,20.8007890763,0.0,0],
[20.8007890763,0.0,22.7551064488,0.0,1],
[22.5551064488,0.2,12.3463820888,0.2,0],
[12.3463820888,0.2,10.5739139352,0.2,2],
[10.5739139352,0.2,1.78257088198,0.2,0],
[1.78257088198,0.2,1.78257088198,-2.42539837758,1],
[1.78257088198,-2.42539837758,4.8901598196,-2.42539837758,0],
[4.8901598196,-2.42539837758,5.33388782367,-2.42539837758,2],
[5.33388782367,-2.42539837758,10.0021387639,-2.42539837758,0],
[10.0021387639,-2.42539837758,12.0585833555,-2.42539837758,3],
[12.0585833555,-2.42539837758,14.0555744687,-2.42539837758,0],
[14.0555744687,-2.42539837758,14.7245656932,-2.42539837758,2],
[14.7245656932,-2.42539837758,22.0483378369,-2.42539837758,0],
[22.0483378369,-2.42539837758,23.5322613972,-2.42539837758,1],
[23.3322613972,-2.22539837758,16.6421787591,-2.22539837758,0],
[16.6421787591,-2.22539837758,16.1953782643,-2.22539837758,2],
[16.1953782643,-2.22539837758,13.5251081963,-2.22539837758,0],
[13.5251081963,-2.22539837758,11.5202847592,-2.22539837758,3],
[11.5202847592,-2.22539837758,7.44967438232,-2.22539837758,0],
[7.44967438232,-2.22539837758,6.99335562487,-2.22539837758,2],
[6.99335562487,-2.22539837758,3.4333370467,-2.22539837758,0],
[3.4333370467,-2.22539837758,3.4333370467,-5.04735996947,1],
[3.4333370467,-5.04735996947,6.09511006139,-5.04735996947,0],
[6.09511006139,-5.04735996947,8.28760347688,-5.04735996947,2],
[8.28760347688,-5.04735996947,11.8729530459,-5.04735996947,0],
[11.8729530459,-5.04735996947,13.6555133822,-5.04735996947,3],
[13.6555133822,-5.04735996947,16.5457517755,-5.04735996947,0],
[16.5457517755,-5.04735996947,16.7705054371,-5.04735996947,2],
[16.7705054371,-5.04735996947,24.0970331597,-5.04735996947,0],
[24.0970331597,-5.04735996947,25.164161016,-5.04735996947,1],
[24.964161016,-4.84735996947,18.242079936,-4.84735996947,0],
[18.242079936,-4.84735996947,17.5812691247,-4.84735996947,2],
[17.5812691247,-4.84735996947,15.1343313517,-4.84735996947,0],
[15.1343313517,-4.84735996947,13.3444662594,-4.84735996947,3],
[13.3444662594,-4.84735996947,9.11816030196,-4.84735996947,0],
[9.11816030196,-4.84735996947,7.9198202405,-4.84735996947,3],
[7.9198202405,-4.84735996947,7.9198202405,-4.18192336873,2],
[7.9198202405,-4.18192336873,7.9198202405,-3.23216440457,0],
[7.7198202405,-3.03216440457,7.7198202405,-3.43814645393,2],
[7.7198202405,-3.43814645393,7.7198202405,-4.5988919956,3],
[7.7198202405,-4.5988919956,7.47298207307,-4.5988919956,2],
[7.47298207307,-4.5988919956,4.98405733293,-4.5988919956,0]], dtype=np.float32)


#model
class NeuralLinearNet(torch.nn.Module):

    def __init__(self, input_size, output_size, hl1_size, hl2_size):

        super(NeuralLinearNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hl1_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hl1_size, hl2_size)
        self.fc3 = torch.nn.Linear(hl2_size, hl3_size)
        self.fc4 = torch.nn.Linear(hl3_size, hl4_size)
        self.fc5 = torch.nn.Linear(hl2_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc5(out)
        return out


Amodel = NeuralLinearNet(input_size, output_size, hl1_size, hl2_size).to(device)

sys.stdout = open('out.txt','wt')
MPObj = torch.load('./saved_model.pt').cuda()
idx = [0, 2, 1, 3, 4]
for z in range(x_test.shape[0] - 1):
        mini = 1023.24142
        node = -1
	for k in xrange(0, z - 1):
		datapoint_1 = x_test[z]
		datapoint_2 = x_test[k]
		datapoint_1 = datapoint_1[idx]
		datapoint_2 = datapoint_2[idx]
		#print('here')
		if(datapoint_1[4] != datapoint_2[4]):
			continue
		if(datapoint_1[0] > datapoint_1[1]):
			datapoint_1[0], datapoint_1[1] = datapoint_1[1], datapoint_1[0]
		if(datapoint_2[0] > datapoint_2[1]):
			datapoint_2[0], datapoint_2[1] = datapoint_2[1], datapoint_2[0]
		#print(np.array(fu[idx]))
		datapoint_1 = torch.from_numpy(np.array(datapoint_1)).to(device)
		datapoint_2 = torch.from_numpy(np.array(datapoint_2)).to(device)
		xx = datapoint_1[:4]
		yy = datapoint_2[:4]
		output_a = MPObj(xx.float())
		output_b = MPObj(yy.float())
		if(torch.dist(output_a.squeeze(), output_b.squeeze()) < mini):
			mini = torch.dist(output_a.squeeze(), output_b.squeeze())
			node = k
	
	if(node != -1 and mini < 3.0):
		print(z + 1, node + 1)


