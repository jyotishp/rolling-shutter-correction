#!/usr/bin/env python

from __future__ import print_function
import progressbar
from multiprocessing import cpu_count, Pool
from utils import *

def func(filename):
	rs_sample = DataSample('../oxford/clean_images/' + filename, (224,224), 150)
	rs_sample.crop()
	rs_sample.generateRSSamples()

if __name__ == '__main__':
	dataset_directory = '../'
	oxford_directory = dataset_directory + 'oxford/clean_images'
	img_size = (224, 224)
	samples_per_img = 150

	bar = progressbar.ProgressBar(
	term_width = 56,
		max_value = len(os.listdir(oxford_directory)) * 150,
		widgets = [
			progressbar.Counter(),
			' ',
			progressbar.Bar('=', '[', ']', '.'),
			' ',
			progressbar.ETA()
			])

	print('Generating RS dataset...')

	with Pool(10) as p:
		for i in p.imap_unordered(func, os.listdir(oxford_directory)):
			print(i)
