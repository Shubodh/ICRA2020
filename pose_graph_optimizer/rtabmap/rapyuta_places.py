from PIL import Image as Image_PIL
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
import re
import cv2

import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class RC(torch.nn.Module):
	def __init__(self):
		super(RC, self).__init__()
		self.resnetX = list(torchvision.models.resnet18(pretrained=True).cuda().children())[:9]
		self.modelA = torch.nn.Sequential(*self.resnetX)
		self.lin = torch.nn.Linear(512, 3)

	def forward(self, x):
		x = self.modelA(x)
		x = x.view(-1, 512)
		x = self.lin(x)
		return x


def naturalSort(l):
	convert = lambda text: int(text) if text.isdigit() else text.lower()
	alphanumKey = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
	return sorted(l, key=alphanumKey)


def dispImg(img):
	cvImg = np.array(img)
	cvImg = cvImg[:, :, ::-1].copy()
	cv2.imshow('Image', cvImg)
	cv2.waitKey(0)


if __name__ == '__main__':
	MPObj = RC().cuda()
	checkpoint = torch.load('/home/cair/backup/rapyuta4/pytorch_model/resnet18_best_aug26_data1plus2_changed_3cat.pth.tar')
	MPObj.load_state_dict(checkpoint['state_dict'])

	fileOut = open("labels.txt", 'w')
	imgs = naturalSort([join(sys.argv[1], f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))])
	
	for img in imgs:
		img = Image_PIL.open(img)
		# dispImg(img)

		transform_compose = transforms.Compose([transforms.Resize([224,224]), 
							transforms.RandomHorizontalFlip(),transforms.ToTensor(), 
							transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		img = transform_compose(img)

		img = Variable(img).cuda()
		img = img.unsqueeze(0)

		predLabel = MPObj(img)
		predLabel = predLabel.data.cpu().numpy().squeeze()
		predLabelProb = np.exp(predLabel)
		predLabelProb = predLabelProb/(np.sum(predLabelProb) + 0.0000000001)

		index = np.argmax(predLabelProb)
		labels = ["Rackspace", "Corridor", "Transition"]
		
		# print(labels[index])
		fileOut.write(labels[index]+"\n")

	fileOut.close()