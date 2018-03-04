import os
import h5py
import pickle
import progressbar
import numpy as np
from keras.preprocessing import image

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

rs_imgs = '../data/oxford/rs_images/'
img_files = os.listdir(rs_imgs)
bar = progress(img_files)
valid_files = []
failed = 0
dataset_file = h5py.File('../data/processed/rs_imgs.h5')
for pos, file in enumerate(img_files):
    try:
        img = image.load_img(rs_imgs + file)
        img = image.img_to_array(img)
        file = file.split('.')[0]
        group = dataset_file.create_group(file)
        group.create_dataset('rs_images', data = img)
        valid_files.append(file)
    except Exception as e:
        failed+=1
    if pos % 50 == 0:
        bar.update(pos)

bar.update(len(img_files))

dataset_file.close()
pickle.dump(valid_files, open('../data/processed/vaild_file_list.p', 'wb'))
print('Failed to load', failed, 'images.')