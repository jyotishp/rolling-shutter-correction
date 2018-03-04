import os
import h5py
import pickle
import progressbar
import numpy as np

def progress(annotation_ids):
	bar = progressbar.ProgressBar(
	term_width = 56,
		max_value = len(annotation_ids),
		widgets = [
			progressbar.Counter(),
			' ',
			progressbar.Bar('=', '[', ']', '.'),
			' ',
			progressbar.ETA()
			])
	return bar

gt_path = './data/oxford/rs_ground_truth/'
gt_files = os.listdir(gt_path)
bar = progress(gt_files)
ground_truth = {}
for pos, file in enumerate(gt_files):
	gt = np.genfromtxt(gt_path + file, delimiter = ',')
	gt = gt[::25, :][:,[1,2]].flatten()
	file = file.split('.')[0]
	ground_truth.update({file: gt})
	if pos % 50 == 0:
		bar.update(pos)

bar.update(len(gt_files))

pickle.dump(ground_truth, open('./data/processed/ground_truth.p', 'wb'))