import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
import glob
import cv2
from os import listdir
from os.path import isfile, join
import csv
from torch.utils.data import Dataset, DataLoader
# Hyper Parameters
num_epochs = 5
batch_size = 100
learning_rate = 0.001


class RSdata(Dataset):
    def __init__(self, datapath, labelpath, transform=None):
        self.datapath = datapath
        self.imagenamelist = listdir(self.datapath)
        self.imagepathlist = [datapath + '/' +  i for i in self.imagenamelist]
        self.labelpath = labelpath
        self.transform = transform
        num = 0
        with open(self.labelpath, 'r') as f:
            num = num+1
            print(num)
            reader = csv.reader(f)
            labellist = list(reader)
        self.label = np.asarray(labellist)
    def __len__(self):
        return len(self.imagenamelist)
    def __getitem__(self, idx):
        im = cv2.imread(self.imagepathlist[idx])
        label = self.label[idx]
        if self.transform:
            im = self.transform(im.astype(float))
        # im =  torch.from_numpy(im.astype(float))
        label =  torch.from_numpy(label.astype(float))

        # im = im.double()
        label = label.double()
        # sample = {'image' : im, 'label' : label}
        return im, label
        # return sample


testloaderT = RSdata(datapath = '/tmp/anjan/datacopy/test/images' , labelpath = '/tmp/anjan/datacopy/test/test_labels.csv', transform=transforms.ToTensor() )
trainloaderT = RSdata(datapath = '/tmp/anjan/datacopy/train/images' , labelpath =  '/tmp/anjan/datacopy/train/train_labels.csv', transform=transforms.ToTensor())
testloader = DataLoader(testloaderT, batch_size= batch_size,shuffle=True)
trainloader = DataLoader(trainloaderT, batch_size= batch_size,shuffle=True)


class VanillaCNN(nn.Module):
    def __init__(self):
        super(VanillaCNN, self).__init__()
        self.layer1 = nn.Sequential(
        nn.Conv2d(3,32,kernel_size =(11,11)),
        # nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

        self.layer2 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size =(7,7)),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

        self.layer3 = nn.Sequential(
        nn.Conv2d(64,64,kernel_size =(5,5)),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,64,kernel_size =(3,3)),
        # nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size = (2,2), stride = (2,2)))

        # Reshape or do a .view(-1,9216) after this

        self.layer5 = nn.Sequential(
        nn.Linear(9216, 1024),
        nn.Tanh(),
        nn.Linear(1024 , 256),
        nn.Hardtanh(),
        # nn.Linear(256 , 30))
        nn.Linear(256 , 15))

    def forward(self, x):
        # print("start")
        out = self.layer1(x)
        # print("layer1 is over")
        # print(out.size())
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(-1,9216)
        out = self.layer5(out)
        return out


torch.cuda.set_device(1)
test = VanillaCNN()
test.cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(test.parameters(), lr=learning_rate)
# dtype = torch.cuda(0).FloatTensor
dtype = torch.FloatTensor

# Training
for epoch in range(num_epochs):
    print("Epoch number ----->" + str(epoch))
    for i, it in enumerate(trainloader):
        # print(it[0].size())
        # print(it[1].size())
        images = it[0]
        labels = it[1]
        # print((images.size()))
        images = Variable(images.type(dtype)).cuda()
        labels = Variable(labels.type(dtype)).cuda()
        # print("conv. to varialbes and cast to dtype")
        # break
        # images = Variable(images)
        # labels = Variable(labels)

        optimizer.zero_grad()
        output = test(images)
        # print("got output")
        # print(output.size())
        # break
        loss = criterion(output, labels)
        # print("got loss")
        loss.backward()
        # print("back")
        optimizer.step()
        # print("done")

        if (i+1) % 10 == 0:
            print(len(trainloader))
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' %(epoch+1, num_epochs, i+1, len(trainloader)//batch_size, loss.data[0]))
