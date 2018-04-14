import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch.utils.data as utils
import glob
import cv2
from os import listdir
from os.path import isfile, join
import csv
from torch.utils.data import Dataset, DataLoader


class AF1(nn.Module):
    def __init__(self):
        super(AF1, self).__init__()

        self.layer1 = nn.Sequential(
        nn.Conv2d(3,16,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.layer2 = nn.Sequential(
        nn.Conv2d(16,32,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.layer3 = nn.Sequential(
        nn.Conv2d(32,64,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.layer4 = nn.Sequential(
        nn.Conv2d(64,128,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.layer5 = nn.Sequential(
        nn.Conv2d(128,256,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.layer6 = nn.Sequential(
        nn.Conv2d(256,512,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())


        self.layer7 = nn.Sequential(
        nn.Linear(8192, 4096),
        nn.ReLU(),
        nn.Dropout2d(p=0))

        self.layer8 = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout2d(p=0))



        # ----------------------------------------------------------------------------------------------------



        self.latlayer1 = nn.Sequential(
        nn.Linear(30, 128),
        nn.ReLU())

        self.latlayer2 = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU())




        # ----------------------------------------------------------------------------------------------------


        self.conclayer1 = nn.Sequential(
        nn.Linear(4352, 4096),
        nn.ReLU())

        self.conclayer2 = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU())

        # ----------------------------------------------------------------------------------------------------

        self.dlayer1 = nn.Sequential(
        nn.ConvTranspose2d(64,256,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.dlayer2 = nn.Sequential(
        nn.ConvTranspose2d(256,128,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.dlayer3 = nn.Sequential(
        nn.ConvTranspose2d(128,64,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.dlayer4 = nn.Sequential(
        nn.ConvTranspose2d(64,32,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.dlayer5 = nn.Sequential(
        nn.ConvTranspose2d(32,16,kernel_size =(3,3), stride=(2,2), padding=1),
        nn.ReLU())

        self.dlayer6 = nn.Sequential(
        nn.ConvTranspose2d(16,2,kernel_size =(3,3), stride=(1,1), padding=1),
        nn.ReLU())


        self.dlayer7 = nn.Upsample(size=(224, 224), mode='bilinear')


    def forward(self, x, y):
        # print("start")

        outx = self.layer1(x)
        outx = self.layer2(outx)
        outx = self.layer3(outx)
        outx = self.layer4(outx)
        outx = self.layer5(outx)
        outx = self.layer6(outx)

        outx = outx.view(-1,8192)
        outx = self.layer7(outx)
        outx = self.layer8(outx)

        # ----------------------------------------------------------------------------------------------------

        outy = self.latlayer1(y)
        outy = self.latlayer2(outy)

        # ----------------------------------------------------------------------------------------------------

        concout = torch.cat((outx, outy), 1)

        # ----------------------------------------------------------------------------------------------------
        concout = self.conclayer1(concout)
        concout = self.conclayer2(concout)
        concout = concout.view(-1,64,8,8)

        # ----------------------------------------------------------------------------------------------------

        out = self.dlayer1(concout)
        out = self.dlayer2(out)
        out = self.dlayer3(out)
        out = self.dlayer4(out)
        out = self.dlayer5(out)
        out = self.dlayer6(out)
        out = self.dlayer7(out)
        # Ideally from 1 to 224
        flow = (out-112.5)/(112.5)
        flowtransformed = flow.permute(0,2,3,1)

        final = F.grid_sample(x, flowtransformed)
        return final

