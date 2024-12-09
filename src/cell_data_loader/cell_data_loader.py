#!/usr/bin/python3

import os,sys,json,csv,cv2,glob,random,re,warnings
import numpy as np
import pickle
from numpy.random import choice,shuffle
from math import floor,ceil,sqrt
from PIL import Image,ImageEnhance
from scipy import ndimage
from scipy.signal import resample
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from .base_dataset import BaseDataset
import torch,torchvision
#torchvision.disable_beta_transforms_warning()
from .util import *


#import openslide
#import czifile
try:
	import slideio
except:
	warnings.warn("""No valid slideio installed, SVS and CZI files cannot 
		be read in -- run `pip install slideio` for support""")

class ImageLabelObject(BaseDataset):
	def __init__(self,
			filename,
			mode="whole",
			dtype="torch",
			gpu_ids="",
			dim=(64,64),
			n_channels = None):
		self.filename = filename
		self.image = None
		self.label = None
		self.boxlabel = None
		self.mode = mode
		self.dtype = dtype
		self.dim = dim
		self.gpu_ids = gpu_ids
		self.n_channels = n_channels
	def im_type(self):
		_,ext = os.path.splitext(self.filename)
		ext = ext.lower()
		if ext in [".png",".jpg",".jpeg",".tiff",".tif"]:
			return "regular"
		elif ext in [".svs"]:
			#pip install aicspylibczi>=3.1.1
			return "svs"
		elif ext in [".czi"]:
			return "czi"
		else:
			raise Exception("Image type unsupported: %s" % ext)
	def get_image(self):
		if self.image is None:
			if self.im_type() == "regular":
				self.image = cv2.imread(self.filename)
			#elif self.im_type() == "svs":
				#n = openslide.OpenSlide(self.filename)
				#levels = n.level_dimensions
				#level=len(levels)-1
				#width0,height0 = levels[level]
				#self.image = n.read_region((0,0),level,(width0,height0))
				#self.image = np.array(self.image)
			elif self.im_type() == "czi" or self.im_type() == "svs":
				#https://towardsdatascience.com/slideio-a-new-python-library-for-reading-medical-images-11858a522059
				#self.image = czifile.imread(self.filename)
				try:
					self.image = slideio.open_slide(self.filename,
						self.im_type().upper())
				except:
					raise Exception(
						"SVS/CZI read on %s failed -- check slideio install" %\
						self.filename
					)
				n_scenes = self.image.num_scenes
				self.image = self.image.get_scene(n_scenes-1)
				self.image = self.image.read_block(
					slices=(0,self.image.num_z_slices)
				)
				self.image = np.squeeze(self.image)
				#self.image = np.moveaxis(self.image,0,-1)
			if self.image.dtype != np.uint8:
				self.image = self.image.astype(np.uint8)
			if self.mode == "whole" and self.im_type() == "regular":
				self.image = cv2.resize(self.image,self.dim[::-1])
				assert(not np.isnan(self.image[0,0,0]))
			elif self.mode == "whole" and self.im_type() in ["czi","svs"]:
				#self.image = resize(self.image,self.dim)
				self.image = cv2.resize(self.image,self.dim[::-1])
			if len(self.image.shape) < 3:
				self.image = np.expand_dims(self.image,axis=2)
			s = self.image.shape
			assert(len(s) == 3)
			if s[0] < s[1] and s[0] < s[2]:
				self.image = np.moveaxis(self.image,0,-1)
			if len(self.image.shape) != 3:
				self.image = np.expand_dims(self.image,axis=2)
				assert(len(self.image.shape) == 3)
			if self.n_channels is not None and \
					self.image.shape[2] != self.n_channels:
				self.image = resample(self.image,
					self.n_channels,axis=2)
			if self.dtype == "torch":
				self.image = torch.tensor(self.image)
				assert(len(self.image.size()) == 3)
			elif self.dtype == "numpy":
				assert(len(self.image.shape) == 3)
			else:
				raise Exception("Unimplemented dtype: %s" % self.dtype)
		return self.image
	def get_cell_box_label(self):
		if self.boxlabel is None:
			self.boxlabel,_ = get_cell_boxes_from_image(self.get_image())
		return self.boxlabel
	def get_orig_size(self):
		if self.dtype == "torch":
			s = self.get_image().size()
		elif self.dtype == "numpy":
			s = self.get_image().shape
		else:
			raise Exception("Unimplemented dtype: %s" % self.dtype)
		return s
	def get_n_channels(self):
		if self.n_channels is not None:
			return self.n_channels
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
	def get_box_label_filename(self):
		name,ext = os.path.splitext(self.filename)
		filename = os.path.join(os.path.dirname(name),
			".%s_boxlabel.csv" % os.path.basename(name))
		return filename
	def read_box_label(self,box_label_file=None):
		if box_label_file is None:
			box_label_file = self.get_box_label_filename()
		assert(os.path.isfile(box_label_file))
		self.boxlabel = read_in_csv(box_label_file)
	def __len__(self):
		if self.mode == "whole":
			return 1
		elif self.mode == "sliced":
			x,y = self.get_scaled_dims()
			return x*y
		elif self.mode == "cell":
			return len(self.get_cell_box_label())
		else:
			raise Exception("Invalid mode: %s" % self.mode)
	def __getitem__(self,index):
		if self.mode == "whole":
			return self.get_image()
		elif self.mode == "sliced":
			x_dim,y_dim = self.get_scaled_dims()
			x = index % (x_dim)
			y = (index // (x_dim)) % (y_dim)
			imslice = self.get_image()[x * self.dim[0]:(x+1)*self.dim[0],
				y * self.dim[1]:(y+1)*self.dim[1],...]
			return imslice
		elif self.mode == "cell":
			if len(self.get_cell_box_label()) == 0:
				#print("len self: %d" % len(self))
				return -1
				#raise Exception("No cells in this image — should not call")
				#warnings.warn("No cells found in %s" % self.filename)
				#return -1
			index = index % len(self.get_cell_box_label())
			x,y,l,w = self.get_cell_box_label()[index]
			return slice_and_augment(self.get_image(),x,y,l,w,
				out_size=self.dim)
			raise Exception("Unimplemented")
		else:
			raise Exception("Invalid mode: %s" % self.mode)

class CellDataloader():#BaseDataset):
	def __init__(self,
		*image_folders,
		label_regex = None,
		label_file = None,
		segment_image = "whole",
		augment_image = True,
		dim = (64,64),
		batch_size = 64,
		verbose = True,
		dtype = "torch",
		gpu_ids = None,
		label_balance = True,
		cell_box_regex = None,
		cell_box_filelist = None,
		n_channels = None,
		channels_first = True,
		match_labels=False,
		normalize=True):
		
		self.verbose = verbose
		self.label_balance = label_balance
		self.segment_image = segment_image.lower()
		self.dim = dim
		self.batch_size = batch_size
		self.dtype = dtype
		self.index = 0
		self.im_index = 0
		self.gpu_ids = gpu_ids
		self.cell_box_regex = cell_box_regex
		self.cell_box_filelist = cell_box_filelist
		self.dtype = dtype
		self.n_channels = n_channels
		self.channels_first = channels_first
		self.augment_image = augment_image
		self.normalize = normalize
		if self.augment_image and self.dtype == "torch":
			import torchvision.transforms.v2 as transforms
			
			self.augment = transforms.Compose([
				transforms.RandomHorizontalFlip(0.5),
				transforms.RandomVerticalFlip(0.5),
				transforms.GaussianBlur(5),
				transforms.ColorJitter(brightness=(0.5,1.5),
					contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1)),
				transforms.RandomResizedCrop(self.dim, antialias=True)])#,
				#transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
			self.augment2 = transforms.Compose([
				transforms.ElasticTransform()])
		"""
		If true, outputs given labels equally. This may repeat images if the
		counts of each don't line up.
		"""
		self.match_labels = match_labels
		"""
		Determines if image folders or file lists exist, then reads them in
		"""
		all_filename_lists = []
		duplicate_test = set()
		for img in image_folders:
			flist = get_file_list(img)
			flist_set = set(flist)
			if len(flist_set.intersection(duplicate_test))>0:
				raise Exception("Intersecting files found between labels")
			duplicate_test = duplicate_test.union(flist_set)
			all_filename_lists.append(flist)
		
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
		if label_file is not None and label_regex is not None:
			raise Exception('Cannot have a label file and regex')
		if label_file is not None:
			if not os.path.isfile(label_file):
				raise Exception("No label file: %s" % label_file)
			self.label_input_format = "List"
		elif label_regex is not None:
			assert(isinstance(label_regex,list) or isinstance(label_regex,str))
			if isinstance(label_regex,list):
				assert(len(label_regex) > 0)
				self.label_regex = [re.compile(_) for _ in label_regex]
				if len(self.label_regex) == 1:
					self.label_regex = self.label_regex[0]
				self.n_labels = len(self.label_regex) + 1
			else:
				self.label_regex = re.compile(label_regex)
				self.n_labels = 1
			self.label_input_format = "Regex"
		elif len(all_filename_lists) > 1:
			self.label_input_format = "Folder"
			self.n_labels = len(all_filename_lists)
		if self.verbose:
			print("Detected label format: %s" % self.label_input_format)
		if self.segment_image == "cell":
			"""
			The case when cells are sliced out of images individually. Requires
			Cellpose.
			"""
			try:
				from cellpose import models,io
			except:
				raise Exception("Cellpose import failed -- need valid "+\
					"cellpose version to use cell segmentation option.")
			if self.cell_box_regex is not None:
				assert(len(all_filename_lists) == 1)
				self.cell_box_regex = re.compile(self.cell_box_regex)
				self.label_input_format = "Cell_Box_Regex"
			elif self.cell_box_filelist is not None:
				assert(len(all_filename_lists) == 1)
				if isinstance(self.cell_box_filelist,str): # One image
					self.cell_box_filelist = [[self.cell_box_filelist]]
				elif isinstance(self.cell_box_filelist,list):
					assert(len(self.cell_box_filelist) > 0)
					if isinstance(self.cell_box_filelist[0],str):
						self.cell_box_filelist = [self.cell_box_filelist]
					assert(isinstance(self.cell_box_filelist[0][0],str))
					for i,l in enumerate(self.cell_box_filelist):
						assert(len(all_filenames_list[i]) == len(l))
						for filename in l:
							if not os.path.isfile(filename):
								raise Exception(
									"File doesn't exist: %s" % filename)
				self.label_input_format = "Cell_Box_Filelist"
		
		"""
		Reads in and determines makeup of image folder
		"""
		
		self.image_objects = []
		for j,all_filename_list in enumerate(all_filename_lists):
			for i,filename in enumerate(all_filename_list):
					if is_image_file(filename):
						imlabel = ImageLabelObject(filename,
									gpu_ids=self.gpu_ids,
									dtype=self.dtype,
									dim=self.dim,
									mode=self.segment_image)
						if self.label_input_format == "Folder":
							imlabel.label = j
						elif self.label_input_format == "Regex":
							imlabel.label = self.__matchitem__(filename)
						elif self.label_input_format == "List":
							imlabel.label = label_file_dict[filename]
						elif self.label_input_format == "Cell_Box_Filelist":
							imlabel.read_box_label(self.cell_box_filelist[j][i])
						elif self.label_input_format == "Cell_Box_Regex":
							raise Exception("Unimplemented")
							imlabel.boxlabel = self.cell_im_regex(filename)
						self.image_objects.append(imlabel)
		random.shuffle(self.image_objects)
		if self.match_labels:
			self.sort_to_match_labels()
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
		self.label_batch = None # [ 0 for _ in range(self.batch_size) ]
		if self.return_labels():
			if self.dtype == "numpy":
				self.label_batch = np.zeros((self.batch_size,self.n_labels))
			elif self.dtype == "torch":
				self.label_batch = torch.zeros((self.batch_size,self.n_labels),
					device = self.gpu_ids)
		sample = self.image_objects[0]
		if n_channels is None:
			self.n_channels = sample.get_n_channels()
		else:
			self.n_channels = n_channels
		if self.normalize and self.dtype == "torch":
			from torchvision.transforms.v2 import Normalize
			self.normalizer = Normalize(
				tuple([0.5 for _ in range(self.n_channels)]),
				tuple([0.5 for _ in range(self.n_channels)]))
		for image_object in self.image_objects:
			image_object.n_channels = self.n_channels
		if self.verbose:
			print("%d Channels Detected" % self.n_channels)
		if self.dtype == "torch":
			self.batch = torch.zeros(self.batch_size,self.dim[0],self.dim[1],
				self.n_channels,device=self.gpu_ids)
		elif self.dtype == "numpy":
			self.batch = np.zeros((self.batch_size,self.dim[0],self.dim[1],
				self.n_channels))
		else:
			raise Exception("Unimplemented dtype: %s" % self.dtype)
	def sort_to_match_labels(self):
		assert(self.match_labels)
		if not self.return_labels():
			warnings.warn("Match labels is set to true but no labels found")
		elif "cell" in self.label_input_format.lower():
			warnings.warn("Match labels is set to true but cell-specific" +\
				" matching unimplemented")
		else:
			assert(self.n_labels) > 1
			label_sorter = {}
			for i in range(self.n_labels):
				label_sorter[i] = []
			label_counter = 0
			image_objects_matched = []
			for obj in self.image_objects:
				label_sorter[obj.label].append(obj)
			while not np.all([label_counter >= len(label_sorter[i])\
				 for i in range(self.n_labels)]):
				for j in range(self.n_labels):
					image_objects_matched.append(
						label_sorter[j][label_counter % len(label_sorter[j])])
					label_counter += 1
			self.image_objects = image_objects_matched
			obj_counts = [0 for _ in range(self.n_labels)]
			for obj in self.image_objects:
				obj_counts[obj.label] += 1
			assert(np.all([o == obj_counts[0] for o in obj_counts]))
			assert(len(self.image_objects) // self.n_labels == obj_counts[0])
	def __matchitem__(self,image_file):
		if self.label_input_format != "Regex":
			raise Exception("""
				Label input format must be regex, is currently %s
				""" % self.label_input_format)
		if isinstance(self.label_regex,list):
			m = 0
			for i,reg in enumerate(self.label_regex):
				if bool(reg.search(image_file)):
					if m > 0:
						warnings.warn(
							("Image file %s matches at"+\
							" least two regular expressions") % image_file)
					m = i
			return m
		else:
			if bool(self.label_regex.search(image_file)):
				return 1
			else:
				return 0
	def __iter__(self):
		return self
	def return_labels(self):
		"""
		Boolean determining whether labels or just the image should be returned
		"""
		if self.label_input_format == "None":
			assert(self.n_labels == 0)
			return False
		elif self.segment_image in ["whole","sliced"]:
			assert(self.n_labels > 0)
			return True
		assert(self.n_labels > 0)
		return True
	def next_im(self):
		"""
		Returns the next single image
		"""
		#self.index = (self.index + 1) #% len(self)
		if self.index >= len(self.image_objects):
			self.index = 0
			self.im_index = 0
			raise StopIteration
		#assert(len(self.image_objects[self.index]) > 0)
		im = self.image_objects[self.index][self.im_index]
		if self.return_labels(): label = self.image_objects[self.index].label
		self.im_index += 1
		while self.im_index >= len(self.image_objects[self.index]) or \
				(len(self.image_objects[self.index]) == 0):
			self.im_index = 0
			self.index += 1
			if self.index >= len(self.image_objects): break
		#if self.augment_image and self.dtype == "torch":
			#imdim = im.size()
			#print("imdim: %s" % str(imdim))
			#im = torch.permute(im,[len(imdim)-1]+list(range(0,len(imdim)-1)))
			#print("imdim: %s" % str(im.size()))
			#im = self.augment(im)
			#im = torch.permute(im,list(range(1,len(imdim))) + [0])
			#print("imdim: %s"%str(im.size()))
		if self.return_labels():
			return im,label
		else:
			return im
	def __next__(self):
		"""
		Returns the next batch of images
		"""
		for i in range(self.batch_size):
			if self.return_labels():
				im,y = self.next_im()
				while isinstance(im,int) and im == -1:
					im,y = self.next_im()
				if self.n_labels == 0:
					raise Exception(
						"Cannot return labels with self.n_labels as 0")
				elif self.n_labels == 1:
					self.label_batch[i,0] = y
				else:
					for j in range(self.n_labels):
						self.label_batch[i,j] = 0
					self.label_batch[i,y] = 1
			else:
				im = self.next_im()
				while isinstance(im,int) and im == -1:
					im = self.next_im()
			assert(not isinstance(im,int))
			if self.dtype == "torch":
				if len(im.size()) == 2 and self.n_channels == 1:
					im = torch.unsqueeze(im,2)
				self.batch[i,...] = torch.unsqueeze(im,0)
				if self.channels_first:
					b = torch.moveaxis(self.batch,-1,1)
				else:
					b = self.batch
				if (self.augment_image and self.n_channels <= 3) or self.normalize:
					if not self.channels_first:
						b = torch.permute(b,[0,-1] + list(range(1,len(b.size())-1)))
					#print("b.size(): %s" % str(b.size()))
					if self.n_channels <= 3 and self.augment:
						b = self.augment(b)
						if self.dim[0] > 32 and self.dim[1] > 32:
							b = self.augment2(b)
					#print("b.size(): %s" % str(b.size()))
					#print([0]+list(range(2,len(b.size()))) + [1])
					if self.normalize:
						b = self.normalizer(b)
					if not self.channels_first:
						b = torch.permute(b,[0]+list(range(2,len(b.size()))) + [1])
					#print("b.size(): %s" % str(b.size()))
					#print("sss")
			elif self.dtype == "numpy":
				if len(im.shape) == 2 and self.n_channels == 1:
					im = np.expand_dims(im,axis=2)
				self.batch[i,...] = np.expand_dims(im,axis=0)
				if self.channels_first:
					b = np.moveaxis(self.batch,-1,1)
				else:
					b = self.batch
			else:
				raise Exception("Unimplemented dtype: %s" % self.dtype)
		if self.return_labels():
			return b,self.label_batch
		else:
			return b
