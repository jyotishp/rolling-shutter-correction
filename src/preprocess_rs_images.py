import os
import h5py
import pickle
import progressbar
import numpy as np
import torchvision.transforms as transforms
import cv2

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
        img = cv2.imread(rs_imgs + file)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        preprocess = transforms.Compose([
                                            transforms.ToTensor(),
                                            normalize
                                        ])
        img = preprocess(img)
        file = file.split('.')[0]
        
        gt_name = file.split('_')
        del gt_name[-1]
        gt_name = '_'.join(gt_name)
        gt_img = cv2.imread('../data/oxford/rs_images_gt/' + gt_name + '.jpg')
        gt_img = np.rollaxis(gt_img, 2)
        
        group = dataset_file.create_group(file)
        group.create_dataset('rs_images', data = img)
        group.create_dataset('gt', data = gt_img)
        
        valid_files.append(file)
    except Exception as e:
        failed+=1
    if pos % 50 == 0:
        bar.update(pos)

bar.update(len(img_files))

dataset_file.close()
pickle.dump(valid_files, open('../data/processed/valid_file_list.p', 'wb'))
print('Failed to load', failed, 'images.')