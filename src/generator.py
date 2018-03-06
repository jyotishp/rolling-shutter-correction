import os
import h5py
import pickle
import progressbar
import numpy as np
import random

class Generator():
	def __init__(self, batch_size = 64,
				dataset_file = '../data/processed/rs_imgs.h5',
				filename_list = '../data/processed/vaild_file_list.p',
				ground_truth = '../data/processed/ground_truth.p',
				validation_split = 0.2):
		self.batch_size = batch_size
		self.dataset = h5py.File(dataset_file, 'r')
		file_list = pickle.load(open(filename_list, 'rb'))
		random.shuffle(file_list)
		total_files = len(file_list)
		self.total_files = len(file_list)
		train_split = 1 - validation_split
		self.training_files = file_list[:int(total_files * train_split)]
		self.validation_files = file_list[int(total_files * train_split):]
		self.ground_truth = pickle.load(open(ground_truth, 'rb'))

	def makeEmptyBatch(self):
		img_batch = np.zeros((self.batch_size, 256, 256, 3))
		gt_batch = np.zeros((self.batch_size, 30))
		return img_batch, gt_batch

	def train(self):
		batch_size_counter = 0
		inputs, outputs = self.makeEmptyBatch()
		while True:
			random.shuffle(self.training_files)
			for sample in self.training_files:
				inputs[batch_size_counter, :] = self.dataset[sample]['rs_images'][:]
				outputs[batch_size_counter, :] = self.ground_truth[sample + '_gt']
				batch_size_counter += 1

				if batch_size_counter == self.batch_size:
					yield [inputs, outputs]
					batch_size_counter = 0
					inputs, outputs = self.makeEmptyBatch()

	def validate(self):
		batch_size_counter = 0
		inputs, outputs = self.makeEmptyBatch()
		while True:
			for sample in self.validation_files:
				inputs[batch_size_counter, :] = self.dataset[sample]['rs_images'][:]
				outputs[batch_size_counter, :] = self.ground_truth[sample + '_gt']
				batch_size_counter += 1
				
				if batch_size_counter == self.batch_size:
					yield [inputs, outputs]
					batch_size_counter = 0
					inputs, outputs = self.makeEmptyBatch()
