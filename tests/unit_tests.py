#!/usr/bin/python
import unittest
import os,sys,time

#from src.cell_data_loader import *
import os,sys,time
import numpy as np
import torch

wd         = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
im_root    = os.path.join(os.path.dirname(wd),'data')
#imfile_svs = os.path.join(im_root,'10447627_1.svs')
imfolder1  = os.path.join(im_root,'4173633_5')
imfolder2  = os.path.join(im_root,'3368914_4_non_tumor')
imfolder3  = os.path.join(im_root,'H6-13_Mars1+dapi')
sys.path.insert(0,os.path.join(wd))
from src.cell_data_loader import *

class TestSimple(unittest.TestCase):
	
	def test_basic_load_torch(self):
		batch_size = 5
		dim = (6,7)
		dataset = CellDataloader(imfolder1,batch_size=batch_size,dim=dim,
			dtype='torch',verbose=False,channels_first=False)
		for i,xy in enumerate(dataset):
			s = xy.size()
			self.assertEqual(s[0],batch_size)
			self.assertEqual(s[1],dim[0])
			self.assertEqual(s[2],dim[1])
			self.assertFalse(torch.isnan(xy[0,0,0,0]))
			break
	def test_basic_load_numpy(self):
		batch_size = 5
		dim = (6,7)
		dataset = CellDataloader(imfolder1,batch_size=batch_size,dim=dim,
			dtype='numpy',verbose=False,channels_first=False)
		for i,xy in enumerate(dataset):
			s = xy.shape
			self.assertEqual(s[0],batch_size)
			self.assertEqual(s[1],dim[0])
			self.assertEqual(s[2],dim[1])
			self.assertFalse(np.isnan(xy[0,0,0,0]))
			break
	def test_different_inputs(self):
		batch_size = 5
		dim = (6,7)
		dataset = CellDataloader(imfolder3,dim=dim,batch_size=batch_size,
			dtype='numpy',verbose=False,channels_first=False)
		for i,xy in enumerate(dataset):
			s = xy.shape
			self.assertEqual(s[0],batch_size)
			self.assertEqual(s[1],dim[0])
			self.assertEqual(s[2],dim[1])
			self.assertFalse(np.isnan(xy[0,0,0,0]))
			break
		dataset = CellDataloader(imfolder1,dim=dim,batch_size=batch_size,
			dtype='torch',verbose=False,channels_first=False)
		for i,xy in enumerate(dataset):
			s = xy.size()
			self.assertEqual(s[0],batch_size)
			self.assertEqual(s[1],dim[0])
			self.assertEqual(s[2],dim[1])
			self.assertFalse(np.isnan(xy[0,0,0,0]))
			break
	
	def test_different_folder_inputs(self):
		dataset = CellDataloader(imfolder1,verbose=False)
		self.assertEqual(dataset.label_input_format,"None")
		dataset = CellDataloader([imfolder1,imfolder2],verbose=False)
		self.assertEqual(dataset.label_input_format,"None")
		dataset = CellDataloader(imfolder2,imfolder3,verbose=False)
		self.assertEqual(dataset.label_input_format,"Folder")
		dataset = CellDataloader([imfolder1],verbose=False)
		self.assertEqual(dataset.label_input_format,"None")
		dataset = CellDataloader([imfolder2,imfolder3],verbose=False,
			dtype="numpy")
		self.assertEqual(dataset.label_input_format,"None")
		for xy in dataset:
			self.assertTrue(isinstance(xy,np.ndarray))
			break
		dataset = CellDataloader([imfolder2],imfolder3,verbose=False)
		self.assertEqual(dataset.label_input_format,"Folder")
		dataset = CellDataloader(imfolder1,label_regex=["Test","Train"],
			verbose=False)
		self.assertEqual(dataset.label_input_format,"Regex")
		for xy in dataset:
			self.assertEqual(len(xy),2)
			break
	def test_cellpose(self):
		batch_size = 5
		dim = (6,7)
		for dtype in ["torch","numpy"]:
			dataset = CellDataloader(imfolder3,dim=dim,batch_size=batch_size,
				segment_image="cell",verbose=False,dtype=dtype,
				channels_first=False)
			for i,xy in enumerate(dataset):
				s = xy.shape
				self.assertEqual(s[0],batch_size)
				self.assertEqual(s[1],dim[0])
				self.assertEqual(s[2],dim[1])
				if i > 10: break
	def test_channel_resample(self):
		batch_size = 5
		dim = (6,7)
		for n_channels in range(1,6,2):
			for dtype in ["torch","numpy"]:
				dataset = CellDataloader(imfolder2,imfolder3,dim=dim,
					batch_size=batch_size,n_channels = n_channels,
					channels_first=False,dtype=dtype,verbose=False)
				for x,y in dataset:
					if dtype == "torch":
						s = x.size()
						s_y = y.size()
					elif dtype == "numpy":
						s = x.shape
						s_y = y.shape
					self.assertEqual(s_y[1],2)
					self.assertEqual(s[0],batch_size)
					self.assertEqual(s[1],dim[0])
					self.assertEqual(s[2],dim[1])
					self.assertEqual(s[3],n_channels)
					break
	def test_example_numpy(self):
		try:
			import tensorflow
			valid_tensorflow = True
		except:
			print("Tensorflow import failed -- skipping test")
			valid_tensorflow = False
		if valid_tensorflow:
			from example_numpy import example_numpy
			example_numpy()
	def test_example_torch(self):
		try:
			import torch
			valid_torch = True
		except:
			print("Torch import failed -- skipping test")
			valid_torch = False
		if valid_torch:
			from example.example_torch import example_torch
			example_torch()
	def test_label_match(self):
		batch_size = 1
		dim = (6,7)
		dataset = CellDataloader(imfolder1,imfolder2,imfolder3,dim=dim,
			batch_size=batch_size,match_labels=True)
		counts = [0,0,0]
		for x,y in dataset:
			counts[int(np.argmax(y))] += 1
		self.assertEqual(counts[0],counts[1])
		self.assertEqual(counts[1],counts[2])
	def test_gpu(self):
		batch_size = 5
		dim = (6,7)
		for dtype in ["torch","numpy"]:
			if dtype == "torch":
				if not torch.cuda.is_available():
					print("No available GPU for torch -- skipping")
					continue
				devices = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
			elif dtype == "numpy":
				continue
			for device in devices:
				dataset = CellDataloader(imfolder1,imfolder2,imfolder3,dim=dim,
					batch_size=batch_size,dtype=dtype,match_labels=True,
					gpu_ids = device.idx)
				for x,y in dataset:
					if dtype == "torch":
						self.assertEqual(device.idx,x.device.index)
						self.assertEqual(device.idx,y.device.index)
						break
					elif dtype == "numpy":
						break
					else:
						break

if __name__ == "__main__":
	unittest.main()
