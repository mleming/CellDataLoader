#!/usr/bin/python3

from src.cell_data_loader import CellDataloader

try:
	import tensorflow as tf
except:
	try:
		from keras import tensorflow as tf
	except:
		raise Exception("Need valid tensorflow")

import numpy
import os

def example_numpy(verbose=True):

	"""
	Replace these folders with whatever folders of cells you may have. Note that
	the test/train folders in this script are the same -- you would need to 
	have a separate train/test set in your own implementation
	"""
	wd = os.path.dirname(os.path.realpath(__file__))
	imfolder1 = os.path.join(wd,'data',
		'3368914_4_non_tumor')
	imfolder2 = os.path.join(wd,'data',
		'4173633_5')

	model = resnet50()
	model = tf.keras.applications.resnet50.ResNet50(
		include_top=True,
		weights='imagenet',
		input_tensor=None,
		input_shape=None,
		pooling=None,
		classes=1000
	)

	# Train

	dataloader_train = CellDataloader(imfolder1,imfolder2,
		dtype="numpy",verbose=False)

	z = None
	for epoch in range(100):
		for image,y in dataloader_train:
			if z is None:
				z_dim = list(y.shape)
				z_dim[1] = 1000 - z_dim[1]
				z = np.zeros(z_dim)
			y = np.concatenate(y,z,axis=1)
			model.fit(image,y)

	# Test

	model.eval()
	dataloader_test = CellDataloader(imfolder1,imfolder2,dtype="torch",
		verbose=False)
	total_images = 0
	sum_accuracy = 0
	for image,y in dataloader_test:
		total_images += image.shape[0]
		y_pred = model(image)
		y_pred = y_pred[:,:y.shape[1]]
		sum_accuracy += torch.sum(torch.argmax(y_pred,axis=1) == \
			torch.argmax(y,axis=1))

	accuracy = sum_accuracy / total_images
	if verbose: print("Final accuracy: %.4f" % sum_accuracy)

if __name__ == "__main__":
	example_numpy()
