#!/usr/bin/python3

from cell_data_loader import CellDataloader
from torchvision.models import resnet50 #, ResNet50_Weights
import torch
import os
import glob

def example_torch(gpu_ids = None,verbose=True):

	"""
	Replace these folders with whatever folders of cells you may have. Note that
	the test/train folders in this script are the same -- you would need to 
	have a separate train/test set in your own implementation
	"""

	wd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	model_folder = os.path.join(wd,'checkpoints')
	os.makedirs(model_folder,exist_ok=True)
	model_file = os.path.join(model_folder,'torch_model.pt')
	
	# Training data folders
	imfolder1_train = os.path.join(wd,'data',
		'3368914_4_non_tumor')
	imfolder2_train = os.path.join(wd,'data',
		'4173633_5')
	
	# Test data folders
	imfolder1_test = os.path.join(wd,'data','3368914_4_non_tumor')
	imfolder2_test = os.path.join(wd,'data','4173633_5')
	
	# Get a pretrained model from torchvision
	if os.path.isfile(model_file):
		if verbose: print("Loading %s" % model_file)
		model = torch.load(model_file)
	else:
		model = resnet50(pretrained=True)

	if gpu_ids is not None:
		model.to(gpu_ids)
	
	# Train
	
	model.train()
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	dataloader_train = CellDataloader(imfolder1_train,imfolder2_train,
		dtype="torch",
		verbose=False, gpu_ids=gpu_ids)
	if verbose: print("Beginning training")
	n_epochs = 500
	for epoch in range(n_epochs):
		l = 0
		c = 0
		for image,y in dataloader_train:
			y_pred = model(image)
			y_pred = y_pred[:,:y.size()[1]]
			loss = loss_fn(y_pred, y)
			l += float(loss)
			c += 1
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		torch.save(model,model_file)
		if verbose:
			print(
				"Epoch {epoch:d}/{n_epochs:d}: loss: {loss:.5f}".format(
					epoch=epoch,n_epochs=e_epochs,loss=l/c)
			)
	# Test
	
	model.eval()

	dataloader_test = CellDataloader(imfolder1_test,imfolder2_test,
		dtype="torch",verbose=False, gpu_ids = gpu_ids)
	total_images = 0
	sum_accuracy = 0
	for image,y in dataloader_test:
		total_images += image.size()[0]
		y_pred = model(image)
		y_pred = y_pred[:,:y.size()[1]]
		sum_accuracy += torch.sum(torch.argmax(y_pred,axis=1) == \
			torch.argmax(y,axis=1))
	accuracy = sum_accuracy / total_images
	if verbose: print("Final accuracy: %.4f" % accuracy)

if __name__ == "__main__":
	example_torch()
