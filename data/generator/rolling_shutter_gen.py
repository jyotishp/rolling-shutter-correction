#!/usr/bin/env python

from __future__ import print_function
import progressbar
from utils import *

if __name__ == '__main__':
	dataset_directory = '../'
	oxford_directory = dataset_directory + 'oxford/clean_images'
	img_size = (256, 256)
	samples_per_img = 150

	bar = progressbar.ProgressBar(
	term_width = 56,
		max_value = len(os.listdir(oxford_directory)),
		widgets = [
			progressbar.Counter(),
			' ',
			progressbar.Bar('=', '[', ']', '.'),
			' ',
			progressbar.ETA()
			])

	print('Generating RS dataset...')
	for pos, filename in enumerate(os.listdir(oxford_directory)[3]):
		rs_sample = DataSample(oxford_directory + '/' + filename, img_size, samples_per_img)
		rs_sample.crop()
		rs_sample.generateRSSamples()
		bar.update((pos+1)*150)
