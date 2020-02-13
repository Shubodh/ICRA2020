import argparse
import os
import shutil
import time
import sys

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
rgb_mean = (0.4914, 0.4822, 0.4465)
rgb_std = (0.2023, 0.1994, 0.2010)
#transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize(rgb_mean, rgb_std)])
transform = transforms.Compose([transforms.Resize([224,224]), transforms.RandomHorizontalFlip(),transforms.ToTensor(), transforms.Normalize(rgb_mean, rgb_std)])
BS = 8
epochs = 2000
LR = 0.001
#lr_all = 0.0001
#lr_fc = 0.01
#momentum_all = 0.9
#weight_decay_all = 1e-4

best_prec1 = 0

resnet18 = 'resnet18'

logger_train = Logger('./logs_aug15/train_data1plus2_trisection_good_distribution')
logger_val = Logger('./logs_aug15/val_data1plus2_trisection_good_distribution')

def main():
    #global best_prec1, lr_all, lr_fc
    global best_prec1
   # model_resnet18 = torchvision.models.resnet18(num_classes=4)
   # model_resnet18 = torch.nn.DataParallel(model_resnet18).cuda()
   # checkpoint = torch.load("/home/shubodh/places365_training/trained_models/trained_models_places10_phase1/resnet18_best_phase1_4classes_unfrozen.pth.tar")
   # start_epoch = checkpoint['epoch']
   # best_prec = checkpoint['best_prec1']
   # model_resnet18.load_state_dict(checkpoint['state_dict'])
   # num_ftrs = model_resnet18.module.fc.in_features
   # model_resnet18.module.fc = torch.nn.Linear(num_ftrs,4)
   # model_resnet18.cuda()
   # print("final layer replaced with 4 neurons")
   # print model_resnet18
   # sys.exit(0)
   # cudnn.benchmark = True


    MPObj = RC().cuda()


    train_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_dataset1plus2_trisection_good_distribution.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/dataset1plus2/', transform=transform)
    test_dataset = GetDataset(csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/test_dataset1plus2_trisection_good_distribution.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/dataset1plus2/', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BS, sampler = ImbalancedDatasetSampler(train_dataset, csv_file='/scratch/shubodh/places365/rapyuta4_classes/csv_data_labels/train_dataset1plus2_trisection_good_distribution.csv', root_dir='/scratch/shubodh/places365/rapyuta4_classes/dataset1plus2/'), shuffle=False, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False, num_workers=4)

    
    #w = torch.Tensor([0.46,0.56,5.42,4.69]).cuda()
    w = torch.FloatTensor([0.94,1.1,16.2,9.6]).cuda() # Weight follows Inverse Frequency ratio
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    
    optimizer = torch.optim.Adam(
    [    
         {"params": MPObj.lin.parameters(), "lr": 0.005},
         {"params": MPObj.modelA.parameters(), "lr": 0.001}
    ],
##    momentum=momentum_all,
##    weight_decay=weight_decay_all
        )
    
    #for epoch in range(start_epoch, epochs): #NOTE: Uncomment this if retraining.
    for epoch in range(epochs):
        #adjust_learning_rate(optimizer, epoch)
        
        #learning rate decay every 30 epochs
        #lr_all = lr_all * (0.1 ** ((epoch - 115) // 30))
        
        # train for one epoch
        train(train_loader, MPObj, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, MPObj, criterion, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': resnet18,
#            'state_dict': model_resnet18.state_dict(),
            'state_dict': MPObj.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, resnet18.lower())

class RC(torch.nn.Module):
    def __init__(self):
        super(RC, self).__init__()
        
        self.resnetX = list(torchvision.models.resnet18(pretrained=True).cuda().children())[:9]
        cnt_true = 0
        cnt_false = 0
        
        '''
        for x in self.resnetX:
            for params in x.parameters():
                params.requires_grad = False
        #print(cnt_true, cnt_false)
        '''
        self.modelA = torch.nn.Sequential(*self.resnetX)
        self.lin = torch.nn.Linear(512, 4)
        #self.dp = torch.nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.modelA(x)
        #x = self.dp(x)
        x = x.view(-1, 512)
        x = self.lin(x)
        return x


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        
        # measure accuracy and record loss
        prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))

        
        
    # TENSORBOARD LOGGING
    # 1. Log scalar values (scalar summary)
    info = { 'loss': losses.avg, 'accuracy': top1.avg }

    for tag, value in info.items():
        logger_train.scalar_summary(tag, value, epoch)

    # 2. Log values and gradients of the parameters (histogram summary)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger_train.histo_summary(tag, value.data.cpu().numpy(), epoch)
        logger_train.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch)

    # 3. Log training images (image summary)
    info = { 'train_images': input_var.view(-1, 28, 28)[:10].cpu().numpy() }

    for tag, input_var in info.items():
        logger_train.image_summary(tag, input_var, epoch) 


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()
    true_list = np.array([])
    pred_list = np.array([])
    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(async=True)
            input_var = torch.autograd.Variable(input).cuda()
            target_var = torch.autograd.Variable(target).cuda()
            
            target_ew = torch.autograd.Variable(target.squeeze()).cuda()
            #print(target_ew.size())
            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            # adding custom validation metrics from sklearn
            target_ew_np = np.asarray(target_ew.data.cpu().numpy())
            #print target_ew_np
            true_list = np.append(true_list, target_ew_np)
            #print true_list
            _, pred_label_value = torch.max(output, 1)
            #print(pred_label_value.size())
            pred_list = np.append(pred_list, pred_label_value.cpu().numpy())
            #print(pred_list)
#            print("the length of pred list is {pred}".format(pred = len(pred_list)))
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))
            
                        

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100  == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))



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
            img_name_A = os.path.join('/scratch/shubodh/places365/rapyuta4_classes/dataset1plus2/', self.landmarks.iloc[idx, 0])
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
    torch.save(state, filename + '_latest_aug21_data1plus2_trisection_good_distribution.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest_aug21_data1plus2_trisection_good_distribution.pth.tar', filename + '_best_aug21_data1plus2_trisection_good_distribution.pth.tar')
 
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
