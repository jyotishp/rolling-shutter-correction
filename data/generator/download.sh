#!/bin/bash

# URLs that need to be downloaded
oxford_images_url='http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz'
oxford_data_groundtruth_url='http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/gt_files_170407.tgz'
rangarajan_test_url='http://www.ee.iitm.ac.in/~ee11d035/cvpr17_test_dataset.zip'
current_directory=`pwd`

# Create a directory for Oxford dataset downloads
mkdir -p ../oxford
cd ../oxford
# Download and extract the dataset
echo 'Downloading Oxford dataset...'
curl -L $oxford_images_url > images.tgz
echo 'Downloading ground truth for Oxford dataset...'
curl -L $oxford_data_groundtruth_url > ground_truth.tgz
echo 'Extracting files...'
mkdir -p images
tar xf images.tgz -C images
mkdir -p ground_truth 
tar xf ground_truth.tgz -C ground_truth

echo 'Copying good and okay images...'
mkdir -p clean_images

good_images=`ls ground_truth | grep -E "(good|ok).txt$"`
for gt_filename in $good_images; do
	while read img_filename; do
		cp images/$img_filename.jpg clean_images/$img_filename.jpg
	done < ground_truth/$gt_filename
done