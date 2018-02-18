#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import cv2
import os
from random import random
import math

def bilinearInterpolation(img, pred_pixel, channels = 3):
	row = pred_pixel[0]
	col = pred_pixel[1]
	row_lower = int(row)
	row_upper = row_lower + 1
	col_lower = int(col)
	col_upper = col_lower + 1

	weights = np.zeros(4)
	weights[0] = ( row_upper - row ) * ( col_upper - col )
	weights[1] = ( row_upper - row ) * ( col - col_lower )
	weights[2] = ( row - row_lower ) * ( col_upper - col )
	weights[3] = ( row - row_lower ) * ( col - col_lower )

	p0 = img[col_lower, row_lower]
	p1 = img[col_upper, row_lower]
	p2 = img[col_lower, row_upper]
	p3 = img[col_upper, row_upper]

	return weights[0]*p0 + weights[1]*p1 + weights[2]*p2 + weights[3]*p3

class DataSample(object):
	"""Object for image and corresponding tx, t"""
	def __init__(self, img_path, img_size, samples_per_img):
		self.img_path = img_path
		self.img = cv2.imread(img_path)
		self.img_size = img_size
		self.samples_per_img = samples_per_img
		self.padding = 50
		try:
			os.mkdir('../oxford/rs_images')
		except Exception as e:
			pass
		self.img_name = os.path.basename(img_path)

	def crop(self):
		img_width = self.img.shape[0]
		img_height = self.img.shape[1]
		add_width = self.img_size[0]/2 + self.padding
		add_height = self.img_size[1]/2 + self.padding
		self.img = self.img[ int( (img_width/2) - add_width ) : int( (img_width/2) + add_width ),
							 int( (img_height/2) - add_height ) : int( (img_height/2) + add_height ) ]

	def generateRSSamples(self):
		# Reserve 1/6 of the generated samples for pure transaltional and rotational
		# motions each
		translation_only_count = rotation_only_count = self.samples_per_img / 6
		# Initialize RS image with zeros
		rs_img = np.zeros(self.img.shape)
		# Ground truth for all 150 generated samples will be stored here
		ground_truth = np.zeros((self.samples_per_img, 6))

		# Iterate for 150 samples
		for sample_count in range(self.samples_per_img):
			# tx --> transaltion vector described by (a1, b1, c1)
			# t --> rotation vector described by (a2, b2, c2)
			a1 = ( random() - 0.5 ) * 6
			b1 = ( random() - 0.5 ) * 12
			a2 = ( random() - 0.5 ) * 0.8
			b2 = ( random() - 0.5 ) * 0.8
			c1 = c2 = 0

			# Generate pure transaltion or rotational RS images
			if sample_count < translation_only_count:
				a2 = b2 = c2 = 0
			elif sample_count > translation_only_count and sample_count < 2*rotation_only_count:
				a1 = b1 = c1 = 0

			ground_truth[sample_count] = [a1, b1, c1, a2, b2, c2]

			# Starting X co-ordinate in RS image
			x_start = self.padding / self.img.shape[0]

			# Define helper functions for rotationa dn translation
			def translation(x):
				return a1 * ((x * 5) ** 2) + b1 * x + c1
			def rotation(x):
				return (a2 * (x ** 2) + b2 * x + c2) * math.pi/8

			# Get value of first pixel
			tx_start_frame = translation(x_start)
			t_start_frame = rotation(x_start)

			for i in range(0, self.img.shape[1]):
				x = (i + 1) / self.img.shape[1]
				tx = translation(x) - tx_start_frame
				t = rotation(x) - t_start_frame
				homography = np.array([
						[ math.cos(t), math.sin(t), tx ],
						[ -math.sin(t), math.cos(t), 0 ],
						[ 0, 0, 1 ]
					])
				inv_homography = np.linalg.inv(homography)

				for j in range(0, self.img.shape[0]):
					target_row = j - self.padding - (self.img_size[0]/2)
					target_col = i - self.padding - (self.img_size[1]/2)
					target_pixel = [target_row, target_col, 1]

					source_row, source_col, _ = np.matmul(inv_homography, target_pixel)
					source_row += self.padding + (self.img_size[0]/2)
					source_col += self.padding + (self.img_size[1]/2)
					try:
						rs_img[i, j] = bilinearInterpolation(self.img, (source_row, source_col))
					except Exception as e:
						continue
			rs_img_crop = rs_img[self.padding:self.padding+self.img_size[0], self.padding:self.padding+self.img_size[1]]
			filename = self.img_name.split('.')[0]
			cv2.imwrite('../oxford/rs_images/' + filename + '_' + str(sample_count) + '.jpg', rs_img_crop)
		np.savetxt('../oxford/rs_ground_truth/' + filename + '_gt.txt', ground_truth, delimiter=',')
