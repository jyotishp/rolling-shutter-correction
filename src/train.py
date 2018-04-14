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
from AFmodels import AF1
import pickle
import h5py
import random
import progressbar

# Hyper Parameters
num_epochs = 100
batch_size = 128
learning_rate = 0.0001

dtype = torch.cuda.FloatTensor
test = AF1().cuda()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(test.parameters(), lr=learning_rate, betas=(0.9, 0.999))

validation_split = 0

dataset = h5py.File('../data/processed/rs_imgs.h5', 'r')
file_list = pickle.load(open('../data/processed/valid_file_list.p', 'rb'))
random.shuffle(file_list)
total_files = len(file_list)
total_files = len(file_list)
train_split = 1 - validation_split
training_files = file_list[:int(total_files * train_split)]
validation_files = file_list[int(total_files * train_split):]
ground_truth = pickle.load(open('../data/processed/ground_truth.p', 'rb'))

bar = progressbar.ProgressBar(
    term_width = 72,
    max_value = len(training_files),
    widgets = [
        progressbar.Counter(),
        ' ',
        progressbar.Bar('=', '[', ']', '.'),
        ' ',
        progressbar.ETA(),
        '     ',
        progressbar.DynamicMessage('Loss')
        ])

def makeEmptyBatch():
    img_batch = np.zeros((batch_size, 3, 224, 224))
    gt_batch = np.zeros((batch_size, 30))
    gt_img = np.zeros((batch_size, 3, 224, 224))
    return img_batch, gt_batch, gt_img

prev_loss = 10000000

def train(prev_loss, inputimagesbatch, inputtrajbatch, gtimagesbatch):
    # Inputs
    imagesvar = Variable(inputimagesbatch.type(dtype), requires_grad=False)
    trajvar = Variable(inputtrajbatch.type(dtype), requires_grad=False)

    #outputs
    gtimagesvar = Variable(gtimagesbatch.type(dtype), requires_grad=False)

    img_input = imagesvar.cuda()
    motion_input = trajvar.cuda()
    img_batch = gtimagesvar.cuda()


    optimizer.zero_grad()

    output = test(img_input, motion_input)

    loss = criterion(output, img_batch)
    loss.backward()
    optimizer.step()
    return float(loss)

for i in range(num_epochs):
    print('Starting Epoch:', i)
    batch_size_counter = 0
    pos = 0
    img_input, traj_input, gt_output = makeEmptyBatch()
    random.shuffle(training_files)
    for sample in training_files:
        img_input[batch_size_counter, :] = dataset[sample]['rs_images'][:]
        traj_input[batch_size_counter, :] = ground_truth[sample + '_gt']
        gt_output[batch_size_counter, :] = dataset[sample]['gt'][:]
        batch_size_counter += 1

        if batch_size_counter == batch_size:
            curr_loss = train(
                  prev_loss,
                  torch.from_numpy(img_input), 
                  torch.from_numpy(traj_input), 
                  torch.from_numpy(gt_output)
                 )
            pos += batch_size
            bar.update(pos, Loss = curr_loss)
            batch_size_counter = 0
            img_input, traj_input, gt_output = makeEmptyBatch()
    
    bar.update(len(training_files), Loss = curr_loss)
    print('Loss: ', curr_loss)
    if (prev_loss > curr_loss):
        prev_loss = curr_loss
        torch.save(test, 'model_e' + str(i) + '.pt')
        print('Saved model_e' + str(i))
        
    print('Ended Epoch:', i)
    print('------------------------------------------------')
