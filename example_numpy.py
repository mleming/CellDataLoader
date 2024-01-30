#!/usr/bin/python3

from src.cell_loader_MLEMING import CellDataloader
from keras import tensorflow as tf
import numpy
import os

def example_numpy():
	wd = os.path.dirname(os.path.realpath(__file__))
	imfolder_test = os.path.join(wd,'data','3368914_4_non_tumor')
	imfolder_train = os.path.join(wd,'data','4173633_5')
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

	dataloader_train = CellDataloader(imfolder_train,imfolder_test,
		dtype="numpy",verbose=False)

	for epoch in range(5):
		for image,y in dataloader_train:
			#print("image.size(): %s" % str(image.shape))
			#print("y.size(): %s" % str(y.shape))
			y_pred = y_pred[:,:2]
			#print("y_pred.size(): %s" % str(y_pred.shape))
			model.fit(y_pred,y)

	# Test

	model.eval()

	dataloader_test = CellDataloader(imfolder_test,imfolder_train,dtype="torch",
		verbose=False)
	total_images = 0
	sum_accuracy = 0
	for image,y in dataloader_test:
		total_images += image.shape[0]
		y_pred = model(image)
		sum_accuracy += torch.sum(torch.argmax(y_pred,axis=1) == \
			torch.argmax(y,axis=1))

	accuracy = sum_accuracy / total_images
	#print("Final accuracy: %.4f" % sum_accuracy)

if __name__ == "__main__":
	example_numpy()
