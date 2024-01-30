#!/usr/bin/python3

from src.cell_loader_MLEMING import CellDataloader
from torchvision.models import resnet50
import torch
import os

def example_torch():
	wd = os.path.dirname(os.path.realpath(__file__))
	imfolder_test = os.path.join(wd,'data','3368914_4_non_tumor')
	imfolder_train = os.path.join(wd,'data','4173633_5')
	model = resnet50()

	# Train

	model.train()
	loss_fn = torch.nn.MSELoss(reduction='sum')
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
	dataloader_train = CellDataloader(imfolder_train,imfolder_test,
		dtype="torch",
		verbose=False)

	for epoch in range(5):
		for image,y in dataloader_train:
			#print("image.size(): %s" % str(image.size()))
			#print("y.size(): %s" % str(y.size()))
			y_pred = model(image)
			y_pred = y_pred[:,:2]
			#print("y_pred.size(): %s" % str(y_pred.size()))
			loss = loss_fn(y_pred, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	# Test

	model.eval()

	dataloader_test = CellDataloader(imfolder_test,imfolder_train,dtype="torch",
		verbose=False)
	total_images = 0
	sum_accuracy = 0
	for image,y in dataloader_test:
		total_images += image.size()[0]
		y_pred = model(image)
		sum_accuracy += torch.sum(torch.argmax(y_pred,axis=1) == \
			torch.argmax(y,axis=1))

	accuracy = sum_accuracy / total_images
	#print("Final accuracy: %.4f" % sum_accuracy)

if __name__ == "__main__":
	example_torch()
