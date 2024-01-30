#!/usr/bin/python

import os,sys,json,csv,cv2,glob,random,re
import xml.etree.ElementTree as ET
import numpy as np
import pickle
from numpy.random import choice,shuffle
from math import floor,ceil,sqrt
from PIL import Image,ImageEnhance
from skimage.transform import resize
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from base_dataset import BaseDataset
import torch,torchvision

# Function to distort image, stolen shamelessly from
# https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation
def elastic_transform(im, alpha, sigma, alph_af, rand_s=None):
	"""Elastic deformation of images as described in [Simard2003]_
	(with modifications).
	.. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
		 Convolutional Neural Networks applied to Visual Document
		 Analysis", in Proc. of the International Conference on
		 Document Analysis and Recognition, 2003.
	
	 Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
	"""
	if rand_s is None:
		rand_s = np.random.RandomState(None)
	
	shape = im.shape
	shape_size = shape[:2]
	
	# Random affine
	cent_squ = np.float32(shape_size) // 2
	squ_siz = min(shape_size) // 3
	pts1 = np.float32([cent_squ + squ_siz,
		[cent_squ[0]+squ_siz, cent_squ[1]-squ_siz], cent_squ - squ_siz])
	pts2 = pts1 + rand_s.uniform(-alph_af,
				alph_af, size=pts1.shape).astype(np.float32)
	M = cv2.getAffineTransform(pts1, pts2)
	im = cv2.warpAffine(im, M, shape_size[::-1],
		borderMode=cv2.BORDER_REFLECT_101)
	
	dx = gaussian_filter((rand_s.rand(*shape) * 2 - 1), sigma) * alpha
	dy = gaussian_filter((rand_s.rand(*shape) * 2 - 1), sigma) * alpha
	dz = np.zeros_like(dx)
	
	x, y, z = np.meshgrid(np.arange(shape[1]),\
				np.arange(shape[0]),\
				np.arange(shape[2]))
	indices = np.reshape(y+dy, (-1, 1)),\
				np.reshape(x+dx, (-1, 1)),\
				np.reshape(z, (-1, 1))
	
	return map_coordinates(im,indices,order=1,mode='reflect').reshape(shape)

def slice_and_augment(im,x,y,l,w,
		rotation=0,
		reflection = False,
		color_shift = np.eye(3),
		blur=1.0,
		x_shift = 0.0,
		y_shift = 0.0,
		size_shift = 0.0,
		distort = False,
		contrast = 1.0,
		brightness = 1.0):
	# Random horizontal and vertical shifts
	x = x + round(max(l,w) * x_shift)
	y = y + round(max(l,w) * y_shift)
	
	# Random size shifts
	
	y = int(y - 0.5*(size_shift * w))
	x = int(x - 0.5*(size_shift * l))
	l = int(l * (size_shift + 1))
	w = int(w * (size_shift + 1))
	
	# Arbitrary, continuous rotations
	e = 2# sqrt(2)
	lw = max(w,l)
	rr =  range(floor((y + 0.5 * w) - e * lw),ceil((y + 0.5 * w) + e * lw))
	rr2 = range(floor((x + 0.5 * l) - e * lw),ceil((x + 0.5 * l) + e * lw))
	
	imslice = im.take(rr,axis=0,mode='wrap')
	imslice = imslice.take(rr2,axis=1,mode='wrap')
	
	if reflection:
		imslice = np.flipud(imslice)
	imslice = ndimage.rotate(imslice,rotation)
	
	# Applies elastic distortion; see above	
	if distort:
		# w/ original parameters:
		# imslice = elastic_transform(imslice,
		#	imslice.shape[1] * 3,
		#	imslice.shape[1] * 0.07,
		#	imslice.shape[1] * 0.09)
		imslice = elastic_transform(imslice,
				int(lw/3),
				lw * 0.08,
				lw * 0.04)
				#int(lw/4),
				#lw * 0.07,
				#lw * 0.04)
	
	imshap = imslice.shape
	rr  = range(int(0.5 * imshap[0] - lw/2.0),int(0.5 * imshap[0] + lw/2.0))
	rr2 = range(int(0.5 * imshap[1] - lw/2.0),int(0.5 * imshap[1] + lw/2.0))
	
	imslice = imslice.take(rr,axis=0,mode='wrap')
	imslice = imslice.take(rr2,axis=1,mode='wrap')
	
	# Color shift
	imslice = np.tensordot(imslice,color_shift,(2,1)).astype('uint8')
	
	# Blurring/sharpening (still tinkering with)
	
	#increasing the brightness
	#new_image = PIL.ImageEnhance.Brightness(image).enhance(1.2)
	imslice = Image.fromarray(imslice)
	imslice = ImageEnhance.Brightness(imslice).enhance(brightness)
	imslice = ImageEnhance.Contrast(imslice).enhance(contrast)
	imslice = ImageEnhance.Sharpness(imslice).enhance(blur)
	#imslice = PIL.ImageEnhance.Color(imslice).enhance(color_shift)
	
	#increasing the contrast
	#new_image = PIL.ImageEnhance.Contrast(image).enhance(1.2)
	
	return imslice

def get_augmented_image(im,x,y,l,w,output_im_size = (70,70),augment = False,
	soft_augment = False,c_shift_amt = 0.05,blur_amt = 0.2,
	spatial_shift_amt = 0.02,elastic_distort = True,size_shift = 0.05,
	rotate=True,reflect=True,js_params = None,contrast = 0.2,brightness=0.2):
	
	if js_params is not None:
		for k in js_params:
			p = js_params[k]
			if k == "blur_amt":
				blur_amt = int(p)
			elif k == "spatial_shift_amt":
				spatial_shift_amt = float(p)
			elif k == "elastic_distort":
				elastic_distort = bool(int(p))
			elif k == "rotate":
				rotate = bool(int(p))
			elif k == "reflect":
				reflect = bool(int(p))
			elif k == "c_shift_amt":
				c_shift_amt = bool(int(p))
			elif k == "size_shift":
				size_shift = float(p)
			else:
				raise Exception("Invalid input parameter %s" %k)
		
	c_shift_arr = np.random.rand(3,3)
	c_shift_arr = c_shift_arr / np.sum(c_shift_arr,axis=1)
	c_shift_arr = (np.eye(3) * (1-c_shift_amt) + c_shift_arr * c_shift_amt)
	
	if augment:
		imslice = slice_and_augment(im,x,y,l,w,
		rotation= 0 if not rotate else 360 * np.random.random(),
		reflection=False if not reflect else choice([True,False]),
		color_shift = c_shift_arr,
		blur = 1 + (0.5 - np.random.random()) * 2 * blur_amt,
		contrast = 1 + (0.5 - np.random.random()) * 2 * contrast,
		brightness = 1 + (0.5 - np.random.random()) * 2 * brightness,
		x_shift = (0.5 - np.random.random()) * 2 *spatial_shift_amt,
		y_shift = (0.5 - np.random.random()) * 2 *spatial_shift_amt,
		size_shift = size_shift * (np.random.rand() - 0.5),
		distort = elastic_distort)
	elif soft_augment:
		imslice = slice_and_augment(im,x,y,l,w,
		rotation= 0 if not rotate else 360 * np.random.random(),
		reflection=False if not reflect else choice([True,False]))
	else:
		imslice = slice_and_augment(im,x,y,l,w)
	if not type(imslice) == Image.Image:
		imslice = Image.fromarray(imslice)
	imslice = imslice.resize((output_im_size[0],output_im_size[1]))
	return imslice

def cell_segment_image(image,cache="."):
	return

def read_image_file(imfile,out_format = "torch"):
	basename,ext = os.path.splitext(imfile)
	if ext.lower() == ".jpg":
		return
	elif ext.lower() == ".cz":
		return
	elif ext.lower() == ".svz":
		return
	return

def read_label_file(label_file):
	return

def is_image_file(filename):
	basename,ext = os.path.splitext(filename)
	return ext.lower() in [".jpg",".png",".jpeg",".czs"]

def is_label_file(filename):
	return True

class ImageLabelObject(BaseDataset):
	def __init__(self,
			filename,
			mode="whole",
			dtype="torch",
			gpu_ids="",
			dim=(64,64)):
		self.filename = filename
		self.image = None
		self.label = None
		self.boxlabel = None
		self.mode = mode
		self.dtype=dtype
		self.dim = dim
		self.gpu_ids = gpu_ids
	def get_image(self):
		if self.image is None:
			if self.dtype == "torch":
				cv2_im = cv2.imread(self.filename)
				if self.mode == "whole":
					cv2_im = cv2.resize(cv2_im,self.dim)
				self.image = torch.tensor(cv2_im)
				assert(len(self.image.size()) == 3)
				s = self.image.size()
				if s[0] < s[1] and s[0] < s[2]:
					self.image = self.image.permute(1,2,0)
		return self.image
	def get_orig_size(self):
		if self.dtype == "torch":
			s = self.get_image().size()
		else:
			raise Exception("Unimplemented dtype: %s" % self.dtype)
		return s
	def get_n_channels(self):
		s = self.get_orig_size()
		if len(s) == 2: return 1
		s_sorted = sorted(s,reverse=True)
		return s_sorted[2]
	def get_orig_dims(self):
		s = self.get_orig_size()
		s_sorted = sorted(s,reverse=True)
		x,y = s_sorted[0],s_sorted[1]
		if x == y: return x,y
		if s.index(x) > s.index(y):
			return y,x
		else:
			return x,y
	def get_scaled_dims(self):
		x,y = self.get_orig_dims()
		return x // self.dim[0], y // self.dim[1]
	def __len__(self):
		if self.mode == "whole":
			return 1
		elif self.mode == "sliced":
			x,y = self.get_scaled_dims()
			return x*y
		elif self.mode == "cell":
			raise Exception("Unimplemented")
		else:
			raise Exception("Invalid mode: %s" % self.mode)
	def __getitem__(self,index):
		if self.mode == "whole":
			if self.dtype == "torch":
				return self.get_image()
		elif self.mode == "sliced":
			x_dim,y_dim = self.get_scaled_dims()
			x = index % (x_dim )
			y = (index // (x_dim)) % (y_dim)
			imslice = self.get_image()[x * self.dim[0]:(x+1)*self.dim[0],
				y * self.dim[1]:(y+1)*self.dim[1],...]
			return imslice
		elif self.mode == "cell":
			raise Exception("Unimplemented")
		else:
			raise Exception("Invalid mode: %s" % self.mode)

class CellDataloader():#BaseDataset):
	def __init__(self,
		image_folder,
		label_regex = None,
		label_file = None,
		segment_image = "whole",
		augment_image = True,
		dim = (64,64),
		batch_size = 64,
		verbose = True,
		dtype = "torch",
		gpu_ids = None,
		label_balance = True):
		self.verbose = verbose
		self.label_balance = label_balance
		self.image_folder = image_folder
		self.segment_image = segment_image.lower()
		self.dim = dim
		self.batch_size = batch_size
		self.dtype = dtype
		self.index = 0
		self.im_index = 0
		self.gpu_ids = gpu_ids
		"""
		Determines if image folders exist
		"""
		if isinstance(image_folder,str):
			image_folder = [image_folder]	
		for imf in image_folder:
			if not os.path.isdir(imf):
				raise Exception("Not dir: %s" % imf)
		
		if self.segment_image not in ["whole","cell","sliced"]:
			raise Exception(
			"""
			%s is not a valid option for segment_image.
			Must be 'whole','cell', or 'sliced'
			""" % self.segment_image)

		"""
		Determines the format of the labels in the fed-in data, if they're
		present at all.
		"""
		self.label_input_format = "None"
		self.n_labels = 0
		if self.segment_image in ["whole","sliced"]:
			if label_file is not None and label_regex is not None:
				raise Exception('Cannot have a label file and regex')
			if label_file is not None:
				if not os.path.isfile(label_file):
					raise Exception("No label file: %s" % label_file)
				self.label_input_format = "List"
			elif label_regex is not None:
				assert(isinstance(label_regex,list))
				self.label_regex = [re.compile(_) for _ in label_regex]
				self.label_input_format = "Regex"
			elif len(image_folder) > 1:
				self.label_input_format = "Folder"
			if self.verbose:
				print("Detected label format: %s" % self.label_input_format)
		if self.segment_image == "cell":
			"""
			The case when cells are sliced out of images individually. Requires
			Cellpose.
			"""
			if self.verbose: print("Unimplemented")
		

		"""
		Reads in and determines makeup of image folder
		"""
		self.image_objects = []		
		for i,imf in enumerate(image_folder):
			for root, dirs, files in os.walk(imf, topdown=False):
				for name in files:
					filename = os.path.join(root, name)
					if is_image_file(filename):
						imlabel = ImageLabelObject(filename,
									gpu_ids=self.gpu_ids,
									dtype=self.dtype,
									dim=self.dim,
									mode=self.segment_image)
						if self.label_input_format == "Folder":
							imlabel.label = i
						elif self.label_input_format == "Regex":
							imlabel.label = self.__matchitem__(filename)
						elif self.label_input_format == "List":
							imlabel.label = label_file_dict[filename]
						elif self.label_input_format == "CellImages":
							raise Exception("Unimplemented")
							imlabel.boxlabel = self.cell_im_regex(filename)
						self.image_objects.append(imlabel)
		random.shuffle(self.image_objects)
		if self.verbose:
			print("%d image paths read" % len(self.image_objects))

		"""
		Acts on the above commands to read in the labels as necessary
		"""
		
		if self.label_input_format == "List":
			label_list = read_label_file(label_file)

		"""
		Makes batch array
		"""
		self.batch = None
		self.label_batch = [0 for _ in range(self.batch_size)]
		sample = self.image_objects[0]
		self.n_channels = sample.get_n_channels()
		if self.dtype == "torch":
			self.n_channels = sample.get_n_channels()
			self.batch = torch.zeros(self.batch_size,self.dim[0],self.dim[1],self.n_channels,device=self.gpu_ids)
		elif self.dtype == "numpy":
			raise Exception("Unimplemented dtype: %s" % self.dtype)
		else:
			raise Exception("Unimplemented dtype: %s" % self.dtype)

	#def __len__(self):
	#	return len(self.image_objects)
		
	def __matchitem__(self,image_file):
		if self.label_input_format != "Regex":
			raise Exception("""
				Label input format must be regex, is currently %s
				""" % self.label_input_format)
		m = 0
		for i,reg in enumerate(self.label_regex):
			if bool(reg.search(image_file)):
				if m > 0:
					warnings.warn("""
						Image file %s matches at least two regular expressions
					""" % image_file)
				m = i + 1
		return m
	def __iter__(self):
		return self
	def return_labels(self):
		"""
		Boolean determining whether labels or just the image should be returned
		"""
		if self.label_input_format == "None":
			return False
		elif self.segment_image in ["whole","sliced"]:
			return True
		return True
	def next_im(self):
		"""
		Returns the next single image
		"""
		self.index = (self.index + 1) #% len(self)
		im = self.image_objects[self.index][self.im_index]
		self.im_index += 1
		if self.im_index > len(self.image_objects[self.index]):
			self.im_index = 0
			self.index += 1
		if self.index > len(self.image_objects):
			self.index = 0
			raise StopIteration
		if self.return_labels():
			return im,self.image_objects[self.index].label
		else:
			return im
	def __next__(self):
		"""
		Returns the next batch of images
		"""
		for i in range(self.batch_size):
			if self.return_labels():
				im,y = self.next_im()
				self.label_batch[i] = y
			else:
				im = self.next_im()
			if self.dtype == "torch":
				self.batch[i,...] = torch.unsqueeze(im,0)
			elif self.dtype == "numpy":
				self.batch[i,...] = np.expand_dims(im,axis=0)
			else:
				raise Exception("Unimplemented dtype: %s" % self.dtype)
			if self.return_labels():
				return self.batch,self.label_batch
			else:
				return self.batch
if True:
	wd = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
	imfolder = '/home/mleming/Dropbox (Partners HealthCare)/9. DROSE'
	dataset = CellDataloader([os.path.join(imfolder,'raw_Test'),os.path.join(imfolder,'raw_Train, Val')],segment_image = "whole")
	#dataset = CellDataloader(imfolder,label_regex=["Test","Train"])

	for i,(x,y) in enumerate(dataset):
		print(sys.getsizeof(x))
		#print(i.label)
		#print(i.filename)

