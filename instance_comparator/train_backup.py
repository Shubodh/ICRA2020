import torch
import torchvision
import torch.nn
import numpy as np
import random
import torch.nn.functional as F
#import matplotlib.pyplot as plt

device = 'cuda'

#hyper parameters
input_size = 5
output_size = 8
hl1_size = 3
hl2_size = 3
hl3_size = 32
hl4_size = 16
num_epochs = 800
learning_rate = 0.001

import torch
import torch.nn

'''
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):

class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            print('jajajaja',output1.size(), output2.size())
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contr

'''

class ContrastiveLoss(torch.nn.Module):

      def __init__(self, margin=1.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
           # print('jajaja',output1.size(), output2.size())
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive
'''
#input
x_train = np.array([[3.3, 1.2, 4.2, 2.1], [13.2, 31.2, 12.1, 13.2]], dtype=np.float32)

y_train = np.array([[6.6], [8.8]], dtype=np.float32)
'''

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



'''
def get_training_data():
    x = 10


    
x_train, y_train = get_training_data()
x_test, y_test = get_testing_data()
'''

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
        #out = out.view(out.size(0), -1)
       # out = self.relu(out)
       # out = self.fc3(out)
       # out = self.relu(out)
       # out = self.fc4(out)
       # out = self.relu(out)
        out = self.fc5(out)
        return out


Amodel = NeuralLinearNet(input_size, output_size, hl1_size, hl2_size).to(device)

LOSS_FUNCTION = ContrastiveLoss(1.0).to(device)

#criterion and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Amodel.parameters(), learning_rate)

#Amodel = torch.load('Amodel.ckpt').to(device)

def get_input(c):
	#R-R-1
	if(c == 1):
		ret_1 = []
                ret_3 = []
                ret_2 = [1]
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(1.5, 4.0)
		ba = random.uniform(0, 50)
		bb = ba
		bc = ba + random.uniform(-1.5, 1.5)
		bd = bc
		lc = la + random.uniform(-1.5, 1.5)
		ld = lc + random.uniform(1.5, 4.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(0)
		ret_3.append(0)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
	#R-R-0	
	elif(c == 2):
		#did not code ba=bc=bb=bd case
		ret_1 = []
		ret_3 = []
                ret_2 = [c]
	 	#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(1.5, 4.0)
		ba = random.uniform(0, 50)
		bb = ba
		bc = bb + random.uniform(4.5, 30.5)
		bd = bc
		lc = lb + random.uniform(5.5, 10.5)
		ld = lc + random.uniform(1.5, 4.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(0)
		ret_3.append(0)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
	#C-C-1
	elif(c == 3):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-2.5, 2.5)
		ba = random.uniform(0, 50)
		bb = ba + random.uniform(-19.5, 19.5)
		bc = ba + random.uniform(-10.5, 10.5)
		bd = bc + random.uniform(-19.5, 19.5)
		lc = la + random.uniform(-2.0, 2.0)
		ld = lc + random.uniform(-2.5, 2.5)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(1)
		ret_3.append(1)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
	#C-C-0
	elif(c == 4):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-2.5, 2.5)
		ba = random.uniform(0, 50)
		bb = ba + random.uniform(-19.5, 19.5)
		bc = ba + random.uniform(-10.5, 10.5)
		bd = bc + random.uniform(-19.5, 19.5)
		lc = la + random.uniform(5.5, 20.0)
		ld = lc + random.uniform(-2.5, 2.5)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(1)
		ret_3.append(1)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
	
	#I-I-1	
	elif(c == 5):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-2.0, 2.0)
		ba = random.uniform(0, 50)
		bb = ba 
		bc = ba + random.uniform(-1.4, 1.5)
		bd = bc
		lc = la + random.uniform(-1.5,1.5)
		ld = lc + random.uniform(-2.0, 2.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(3)
		ret_3.append(3)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
		
	#I-I-0	
	elif(c == 6):
#		elif(c == 5):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-2.0, 2.0)
		ba = random.uniform(0, 50)
		bb = ba 
		bc = ba + random.uniform(5.5, 36.5)
		bd = bc
		lc = random.uniform(0,50)
		ld = lc + random.uniform(-2.0, 2.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(3)
		ret_3.append(3)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
		
	#T-T-1	
	elif(c == 7):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-1.0, 1.0)
		ba = random.uniform(0, 50)
		bb = ba 
		bc = ba + random.uniform(-0.5, 0.5)
		bd = bc
		lc = la + random.uniform(-0.5,0.5)
		ld = lc + random.uniform(-1.0, 1.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(2)
		ret_3.append(2)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
		
	#T-T-0	
	elif(c == 8):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		#ret_2 = np.array(ret_2)
		la = random.uniform(0,50)
		lb = la + random.uniform(-1.0, 1.0)
		ba = random.uniform(0, 50)
		bb = ba 
		bc = ba + random.uniform(5.5, 36.5)
		bd = bc
		lc = la + random.uniform(-0.5,0.5)
		ld = lc + random.uniform(-1.0, 1.0)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(2)
		ret_3.append(2)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
    #R-I-0
	elif(c == 9):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		la = random.uniform(0,50)
		lb = la + random.uniform(1.5, 4.0)
		ba = random.uniform(0, 50)
		bb = ba
		lc = random.uniform(0,50)
		ld = lc + random.uniform(-2.0, 2.0)
		bc = random.uniform(0, 50)
		bd = bc
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(0)
		ret_3.append(3)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))

    #R-C-0
	elif(c == 10):
		ret_1 = []
		ret_2 = [c]
                ret_3 = []
		la = random.uniform(0,50)
		lb = la + random.uniform(1.5, 4.0)
		ba = random.uniform(0, 50)
		bb = ba
		lc = random.uniform(0,50)
               	ld = lc + random.uniform(-2.5, 2.5)
		bc = random.uniform(0, 50)
		bd = bc + random.uniform(-19.5, 19.5)
		ret_1.append(la)
		ret_1.append(lb)
		ret_1.append(ba)
		ret_1.append(bb)
		ret_3.append(lc)
		ret_3.append(ld)
		ret_3.append(bc)
		ret_3.append(bd)
		ret_1.append(0)
		ret_3.append(1)
		return(np.array(ret_1), np.array(ret_2), np.array(ret_3))
	
	
#MLP to check node equality.
#start training

for epoch in range(num_epochs):
	
    #inputs = torch.from_numpy(x_train).to(device)
    #labels = torch.from_numpy(y_train).to(device)

    
    c = 1
    loss = 0
    zz = 0
    for j in range(1000):
                #print(j)
	        x_inp, y_inp, x_inp_2 = get_input(c)
		x_inp = torch.from_numpy(x_inp).to(device)
                x_inp_2 = torch.from_numpy(x_inp_2).to(device)
		y_inp = torch.from_numpy(y_inp).to(device)
		output_a = Amodel(x_inp.float())
                zz = zz + 1
                #if(zz % 10 == 0):
                #    print('hey', zz)
                output_b = Amodel(x_inp_2.float())
                output_a = output_a.unsqueeze(0)
                output_b = output_b.unsqueeze(0)
               # print('hey', output_a.size(), output_b.size())
		loss += LOSS_FUNCTION(output_a, output_b, y_inp.float())
		c = c + 1
		if(c == 11):
			c = 1
			    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch:', epoch,  'loss:', loss.item())

    if(epoch % 10 == 0):
    	print("saving model at epoch {}".format(epoch))
    	torch.save(Amodel, './saved_ckpt/latest_save_{0:05d}.pt'.format(epoch))
print('hw')
torch.save(Amodel, 'Amodel_mlp.ckpt')
'''
#10.5739139352,0.2,1.78257088198,0.2,0
MPObj = torch.load('./Amodel_mlp.ckpt').cuda()
z = [14.7245656932, 22.0483378369,-2.42539837758,-2.42539837758,16.6421787591,23.3322613972,-2.22539837758,-2.22539837758, 0,0]
z, ffuu = get_input(9)
x_inp = torch.from_numpy(z).to(device)
print(x_inp)
op = MPObj(x_inp.float())
print(op)
'''

