import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from logger import Logger

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

import pandas as pd
import wideresnet
import pdb
import numpy as np
from PIL import Image
#seed = 42
#np.random.seed(seed)
#torch.manual_seed(seed)
#save_every = 5
#rgb_mean = (0.4914, 0.4822, 0.4465)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#rgb_std = (0.2023, 0.1994, 0.2010)
transform = transforms.Compose([transforms.Resize([224,224]), transforms.RandomHorizontalFlip(),transforms.ToTensor()])

# Hyper-parameters
sequence_length = 224 #28
input_size = 224
hidden_size = 32 #128
num_layers = 2
num_classes = 4
batch_size = 100 #100
num_epochs = 20
learning_rate = 0.01 #0.003

best_prec1 = 0

logger_train = Logger('./logs_bid_lstm_1/train_1')
logger_val = Logger('./logs_bid_lstm_1/val_1')

def main():
    global best_prec1, sequence_length, input_size, hidden_size, num_layers, num_classes, batch_size, num_epochs, learning_rate 
    model = BiRNN(input_size, hidden_size, num_layers, num_classes).to(device)
    
    train_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/train_all/', transform=transform)
    test_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/test.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/test_6900_7900/', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler = ImbalancedDatasetSampler(train_dataset, csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_all.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/train_all/'), shuffle=False, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(train_loader)
    for epoch in range(num_epochs):
        #adjust_learning_rate(optimizer, epoch)
        #learning rate decay every 30 epochs
        #lr_all = lr_all * (0.1 ** ((epoch - 115) // 30))
        
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        #print optimizer.param_groups
        # evaluate on validation set
        #prec1 = validate(val_loader, model, criterion, epoch)

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model_1.ckpt')
        # remember best prec@1 and save checkpoint
       # is_best = prec1 > best_prec1
       # best_prec1 = max(prec1, best_prec1)
       # save_checkpoint({
       #     'epoch': epoch + 1,
       #     'arch': resnet18,
       #     'state_dict': model_resnet18.state_dict(),
       #     'best_prec1': best_prec1,
       # }, is_best, resnet18.lower())


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
#    for i, (input, target) in enumerate(train_loader):
#        target = target.cuda(async=True)
#        input_var = torch.autograd.Variable(input)
#        target_var = torch.autograd.Variable(target)

    for i, (images, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        print(images.shape)
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # measure accuracy and record loss
        prec1, prec3 = accuracy(outputs.data, labels, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i+1) % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))
        
    # TENSORBOARD LOGGING
#    # 1. Log scalar values (scalar summary)
#    info = { 'loss': losses.avg, 'accuracy': top1.avg }
#
#    for tag, value in info.items():
#        logger_train.scalar_summary(tag, value, epoch)
#
#    # 2. Log values and gradients of the parameters (histogram summary)
#    for tag, value in model.named_parameters():
#        tag = tag.replace('.', '/')
#        logger_train.histo_summary(tag, value.data.cpu().numpy(), epoch)
#        logger_train.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
   # info = { 'train_images': input_var.view(-1, 28, 28)[:10].cpu().numpy() }

   # for tag, input_var in info.items():
   #     logger_train.image_summary(tag, input_var, epoch) 


def validate(val_loader, model, criterion, epoch):
#    batch_time = AverageMeter()
#    losses = AverageMeter()
#    top1 = AverageMeter()
#    top3 = AverageMeter()
#    true_list = np.array([])
#    pred_list = np.array([])
#    # switch to evaluate mode
#    model.eval()
#
#    end = time.time()
#    with torch.no_grad():
#        for i, (input, target) in enumerate(val_loader):
#            target = target.cuda(async=True)
#            input_var = torch.autograd.Variable(input)
#            target_var = torch.autograd.Variable(target)
#            
#            target_ew = torch.autograd.Variable(target.squeeze()).cuda()
#            #print(target_ew.size())
#            # compute output
#            output = model(input_var)
#            loss = criterion(output, target_var)
#            # adding custom validation metrics from sklearn
#            target_ew_np = np.asarray(target_ew.data.cpu().numpy())
#            #print target_ew_np
#            true_list = np.append(true_list, target_ew_np)
#            #print true_list
#            _, pred_label_value = torch.max(output, 1)
#            #print(pred_label_value.size())
#            pred_list = np.append(pred_list, pred_label_value.cpu().numpy())
#            #print(pred_list)
##            print("the length of pred list is {pred}".format(pred = len(pred_list)))
#            # measure accuracy and record loss
#            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
#            losses.update(loss.item(), input.size(0))
#            top1.update(prec1.item(), input.size(0))
#            top3.update(prec3.item(), input.size(0))
#
#            # measure elapsed time
#            batch_time.update(time.time() - end)
#            end = time.time()
#
#            if i % 10  == 0:
#                print('Test: [{0}/{1}]\t'
#                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
#                       i, len(val_loader), batch_time=batch_time, loss=losses,
#                       top1=top1, top3=top3))


    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total)) 

    # TENSORBOARD LOGGING
    # 1. Log scalar values (scalar summary)
    info = { 'loss': losses.avg, 'accuracy': top1.avg }

    for tag, value in info.items():
        logger_val.scalar_summary(tag, value, epoch)

    print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
          .format(top1=top1, top3=top3))
   # accuracy score, confusion_matrix and classification report from sklearn
    print('Confusion matrix: ')
    print confusion_matrix(true_list, pred_list)
    print('Accuracy score: ', accuracy_score(true_list, pred_list))
    print(classification_report(true_list,pred_list))

    return top1.avg


# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size #128
        self.num_layers = num_layers #2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, csv_file, root_dir, indices=None, num_samples=None):
        #print('here')        
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
            
        # distribution of classes in the dataset 
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
                
        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
        #print('here111')
        
        
    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            img_name_A = os.path.join('/scratch/shubodh/places365/rapyuta4_classes/train_all/', self.landmarks.iloc[idx, 0])
            label = self.landmarks.iloc[idx, 1] 
            #print(img_name_A, label)
            return (label)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

class GetDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.landmarks = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        img_name_A = os.path.join(self.root_dir, self.landmarks.iloc[idx, 0])
        label = self.landmarks.iloc[idx, 1] 
        #print(self.landmarks.iloc[idx, 0], label)
        #time.sleep(1)
        return (self.transform(Image.open(img_name_A)), label)




def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_all = lr_all * (0.1 ** ((epoch - 117) // 30))
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = lr



def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')
 
    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

def calculateTotalLoss(targ, preda):
   
    w = torch.Tensor([0.2,0.6,1.8,0.6]).cuda()
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    return criterion(preda,targ)
    
    


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


#write object for tensorboard.
#writer = SummaryWriter('/home/tourani/Desktop/region-classification-matterport-pytorch/lf')

#MPObj = torch.load('/scratch/satyajittourani/saved_models_raputa_resize/latest_saved_state_005.pt').cuda()
if __name__ == '__main__':
    main()
