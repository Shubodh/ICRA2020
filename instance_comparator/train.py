import torch
import torchvision
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
device = 'cuda'

#hyper parameters
input_size = 2
output_size = 1
hl1_size = 600
hl2_size = 600
hl3_size = 700
hl4_size = 700
num_epochs = 1000
learning_rate = 0.001


#input
#x_train = np.array([[1.3,3.5], [22.3,1.2], [2.3, 4.6], [12.0, 1.0], [1.1, 0.6], [180.0, 7.0], [2.3, 4.4], []], dtype=np.float32)

#y_train = np.array([[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0],[1], [0]], dtype=np.float32)


def get_data():
   x_tr = []
   y_tr = []
   x_ts = []
   y_ts = []
   
   xr = 0;
   
   for i in xrange(0, 100, 1):
       for j in xrange(0, 100, 1):
           xr = xr + 1
           if(xr % 4 == 0):
               x_ts.append([i, j])
               if(i < 5 and j < 5):
                   y_ts.append(1)
               else:
                   y_ts.append(0)
           else:
               x_tr.append([i, j])
               if(i < 5 and j < 5):
                   y_tr.append(1)
               else:
                   y_tr.append(0)
   return np.array(x_tr, dtype=np.float32),np.array(y_tr, dtype=np.float32), np.array(x_ts, dtype=np.float32), np.array(y_ts, dtype=np.float32)
    
x_train, y_train, x_test, y_test  = get_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

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
       # out = self.relu(out)
       # out = self.fc3(out)
       # out = self.relu(out)
       # out = self.fc4(out)
       # out = self.relu(out)
        out = self.fc5(out)
        return out


Amodel = NeuralLinearNet(input_size, output_size, hl1_size, hl2_size).to(device)

#criterion and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(Amodel.parameters(), learning_rate)

#Amodel = torch.load('Amodel.ckpt').to(device)

#start training
for epoch in range(num_epochs):
    inputs = torch.from_numpy(x_train).to(device)
    labels = torch.from_numpy(y_train).to(device)
    output = Amodel(inputs)
    output = output.squeeze()
    loss = criterion(output, labels)

    #backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch:', epoch,  'loss:', loss.item())
print('hw')
torch.save(Amodel, 'Amodel.ckpt')

tot = 0
corr = 0
get_labels = []
xr = 0
for i in range(len(x_test)):
    inputs = torch.from_numpy(x_test[i]).to(device)
    yo = y_test[i]
    yo = np.array(yo)
    labels = torch.from_numpy(yo).to(device)
    output = Amodel(inputs)
    output = output.squeeze()
    xr = xr + 1
    if(xr == 0 and y_test[i] == 0):
        get_labels.append(output.data[0].cpu().numpy())
    if(xr == 1 and y_test[i] == 1):
		get_labels.append(output.data[0].cpu().numpy())
    
    if(abs(output.data[0].cpu().numpy() - y_test[i]) < 0.1):
		corr = corr + 1
    tot = tot + 1
    loss = criterion(output, labels)
    
    #backward
    optimizer.zero_grad()
    #loss.backward()
    #optimizer.step()

get_labels = np.array(get_labels)
ho = zip(get_labels, y_test)
random.shuffle(ho)
print(ho[0:200])
print(corr, tot)
