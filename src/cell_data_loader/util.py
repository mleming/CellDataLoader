#!/usr/bin/python

import os,sys,json,csv,cv2,glob,random,re
import numpy as np
from numpy.random import choice,shuffle
from math import floor,ceil,sqrt
from PIL import Image,ImageEnhance
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from .base_dataset import BaseDataset
import torch,torchvision
from contextlib import contextmanager
import warnings
#from random import shuffle
try:
	from cellpose import models, io
except:
	warnings.warn("Working Cellpose not installed -- cell segmentation not"+\
	" possible")

import skimage
# DEFINE CELLPOSE MODEL

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

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
		trans = None,
		out_size = None,
		rotation=0,
		x_shift = 0.0,
		y_shift = 0.0,
		size_shift = 0.0):
	"""
	Slices out and augments a particular cell from a larger image, given
	the box coordinates (x,y,l <length>, w <width>). The transformation array
	can be added to augment images as well. The rotation and shifts are applied
	separately since they may need to consider outside images for the corners.
	"""
	if isinstance(im,np.ndarray):
		dtype = "numpy"
		imsize = im.shape
	elif torch.is_tensor(im):
		dtype = "torch"
		imsize = im.size()
	else:
		raise Exception("Unimplemented type in slice_and_augment")
	# Random horizontal and vertical shifts
	x = x + round(max(l,w) * x_shift)
	y = y + round(max(l,w) * y_shift)
	
	# Random size shifts
	
	y = int(y - 0.5*(size_shift * w))
	x = int(x - 0.5*(size_shift * l))
	l = int(l * (size_shift + 1))
	w = int(w * (size_shift + 1))
	
	# Arbitrary, continuous rotations
	e   = 2 # sqrt(2)
	lw  = max(w,l)
	rr  = range(floor((y + 0.5 * w) - e * lw),ceil((y + 0.5 * w) + e * lw))
	rr  = [_ % imsize[0] for _ in rr]
	rr2 = range(floor((x + 0.5 * l) - e * lw),ceil((x + 0.5 * l) + e * lw))
	rr2 = [_ % imsize[1] for _ in rr2]
	
	if dtype == "torch":
		imslice = torch.index_select(im,0,torch.LongTensor(rr))
		imslice = torch.index_select(imslice,1,torch.LongTensor(rr2))
	elif dtype == "numpy":
		imslice = im.take(rr,axis=0)#,mode='wrap')
		imslice = imslice.take(rr2,axis=1)#,mode='wrap')
	else:
		raise Exception("Unimplemented dtype: %s" % dtype)
	
	if dtype == "torch":
		imshap = imslice.size()
	elif dtype == "numpy":
		imshap = imslice.shape
	else:
		raise Exception("Unimplemented dtype: %s" % dtype)

	if dtype == "torch":
		imslice = torchvision.transforms.functional.rotate(imslice,rotation)
	elif dtype == "numpy":
		imslice = ndimage.rotate(imslice,rotation)
	else:
		raise Exception("Unimplemented dtype: %s" % dtype)
	
	# Applies elastic distortion; see above	
	rr  = range(int(0.5 * imshap[0] - lw/2.0),int(0.5 * imshap[0] + lw/2.0))
	rr  = [_ % imshap[0] for _ in rr]
	rr2 = range(int(0.5 * imshap[1] - lw/2.0),int(0.5 * imshap[1] + lw/2.0))
	rr2 = [_ % imshap[1] for _ in rr2]
	
	if dtype == "torch":
		imslice = torch.index_select(imslice,0,torch.LongTensor(rr))
		imslice = torch.index_select(imslice,1,torch.LongTensor(rr2))
	elif dtype == "numpy":
		imslice = imslice.take(rr,axis=0)#,mode='wrap')
		imslice = imslice.take(rr2,axis=1)#,mode='wrap')
	
	#imslice = Image.fromarray(imslice)
	if trans is not None: imslice = trans(imslice)
	
	#imslice = PIL.ImageEnhance.Color(imslice).enhance(color_shift)
	
	#increasing the contrast
	#new_image = PIL.ImageEnhance.Contrast(image).enhance(1.2)
	if out_size is not None:
		if dtype == "torch":
			imslice = torch.permute(torchvision.transforms.Resize(
				out_size)(torch.permute(imslice,(2,0,1))),(1,2,0))
		elif dtype == "numpy":
			imslice = cv2.resize(imslice,out_size[::-1])
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
	return ext.lower() in [".jpg",".png",".jpeg",".czi",".svs",".tiff",".tif"]

def is_label_file(filename):
	return True

def get_verify(X,Y):
	h = int(X.shape[1] / 2)
	w = int(X.shape[2] / 2)
	q1 = np.expand_dims(np.mean(X[:,:h,:w,:],axis=(1,2,3)),axis=1) / 255.0
	q2 = np.expand_dims(np.mean(X[:,h:,:w,:],axis=(1,2,3)),axis=1) / 255.0
	q3 = np.expand_dims(np.mean(X[:,:h,w:,:],axis=(1,2,3)),axis=1) / 255.0
	q4 = np.expand_dims(np.mean(X[:,h:,w:,:],axis=(1,2,3)),axis=1) / 255.0
	c1 = np.expand_dims(np.mean(X[:,:,:,0],axis=(1,2)),axis=1) / 255.0
	c2 = np.expand_dims(np.mean(X[:,:,:,1],axis=(1,2)),axis=1) / 255.0
	c3 = np.expand_dims(np.mean(X[:,:,:,2],axis=(1,2)),axis=1) / 255.0
	s  = np.expand_dims(np.std(X,axis=(1,2,3)),axis=1) / 255.0
	Y_ =  np.concatenate((Y,q1,q2,q3,q4,c1,c2,c3,s),axis=1)

	return Y_

def tile_image(im,autoencoder,a_dims=(600,600)):
	for i in range(0,im.shape[0],a_dims[0]):
		for j in range(0,im.shape[1],a_dims[1]):
			if i+a_dims[0] > im.shape[0]:
				i = im.shape[0] - a_dims[0] - 1
			if j+a_dims[1] > im.shape[1]:
				j = im.shape[1] - a_dims[1] - 1
			input_im = torch.from_numpy(im[i:i+a_dims[0],
				j:j+a_dims[1],:]).permute(2,1,0).unsqueeze(0).float()/255.0
			input_im = autoencoder(input_im)
			input_im = input_im.squeeze().permute(1,2,0).detach().numpy() * 255.0
			im[i:i+a_dims[0],j:j+a_dims[1],:] = input_im
	return im

def get_all_perimeters(mask):
	perims = {}
	if len(mask.shape) == 2:
		for i in np.unique(mask.flatten()):
			perims[i] = 0
		for i in range(mask.shape[0]-1):
			for j in range(mask.shape[1]-1):
				if mask[i,j] != mask[i+1,j]:
					perims[mask[i,j]] += 1
					perims[mask[i+1,j]] += 1
				if mask[i,j] != mask[i,j+1]:
					perims[mask[i,j]] += 1
					perims[mask[i,j+1]] += 1
	elif len(mask.shape) == 3:
		for i in range(mask.shape[2]):
			perims[i] = 0
	return perims

def get_perimeter(mask):
	p = 0
	for i in range(mask.shape[0]-1):
		for j in range(mask.shape[1]-1):
			if mask[i,j] != mask[i+1,j]:
				p += 1
			if mask[i,j] != mask[i,j+1]:
				p += 1
	return p

def get_smoothness(mask,mean_area = None):
	area = np.sum(mask)
	#return np.abs(area - mean_area)
	return float(skimage.measure.perimeter(mask,neighbourhood=4) ** 2)/area

def crop2(dat, clp=True):
    if clp: np.clip( dat, 0, 1, out=dat )
    for i in range(dat.ndim):
        dat = np.swapaxes(dat, 0, i)  # send i-th axis to front
        while np.all( dat[0]==0 ):
            dat = dat[1:]
        while np.all( dat[-1]==0 ):
            dat = dat[:-1]
        dat = np.swapaxes(dat, 0, i)  # send i-th axis to its original position
    return dat

def remove_unsmooth(mask,thresh=25):
	smoothnesses = np.zeros((len(np.unique(mask))-1,))
	if len(mask.shape) == 2:
		for z in np.unique(mask.flatten()):
			if z > 0:
				z_mask = crop2(mask == z)
				smoothnesses[int(z)-1] = get_smoothness(z_mask)
	elif len(mask.shape) == 3:
		for z in range(mask.shape[2]):			
			smoothnesses[z] = get_smoothness(crop2(np.squeeze(mask[:,:,z])))
	max_smoothness = smoothnesses.max()
	min_smoothness = smoothnesses.min()
	for i in range(mask.max()):
		if smoothnesses[i] > thresh:
			mask[mask == i+1] = 0
	mask = renumber_masks(mask)
	return mask


def get_box_intersection_area(bb1,bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x1, y1) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner
	bb2 : dict
		Keys: {'x1', 'x2', 'y1', 'y2'}
		The (x, y) position is at the top left corner,
		the (x2, y2) position is at the bottom right corner
	
	Returns
	-------
	float
		in [0, 1]
	-----
	Adapted from https://stackoverflow.com/questions/25349178/
	calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
	"""
	# determine the coordinates of the intersection rectangle
	bb1_x1 = bb1[1]
	bb1_x2 = bb1[1] + bb1[3]
	bb1_y1 = bb1[0]
	bb1_y2 = bb1[0] + bb1[2]
	
	bb2_x1 = bb2[1]
	bb2_x2 = bb2[1] + bb2[3]
	bb2_y1 = bb2[0]
	bb2_y2 = bb2[0] + bb2[2]
	
	x_left = max(bb1_x1, bb2_x1)
	y_top = max(bb1_y1, bb2_y1)
	x_right = min(bb1_x2, bb2_x2)
	y_bottom = min(bb1_y2, bb2_y2)
	
	if x_right < x_left or y_bottom < y_top:
		return 0.0
	
	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)
	
	# compute the area of both AABBs
	bb1_area = bb1[3] * bb1[2]
	bb2_area = bb2[3] * bb2[2]

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou

def find_box_intersections(annotated_boxes,predicted_boxes,min_dice=0.5):
	assert(min_dice >= 0.0)
	assert(min_dice <= 1.0)
	final_boxes = []
	for i in range(predicted_boxes.shape[0]):
		append = True
		for j in range(annotated_boxes.shape[0]):
			bb1 = predicted_boxes[i,:]
			bb2 = annotated_boxes[j,:]
			dice = get_box_intersection_area(bb1,bb2)
			if dice > min_dice:
				append = False
		if append:
			final_boxes.append(predicted_boxes[i,:])
	return np.array(final_boxes)

def merge_predicted_and_annotated(annotated_boxes,predicted_boxes,min_dice=0.5):
	intersected_boxes = find_box_intersections(annotated_boxes,predicted_boxes,
		min_dice=min_dice)
	if intersected_boxes.shape[1] == 4:
		neg_labels = np.array(["negative" for _ in range(intersected_boxes.shape[0])])
		neg_labels = np.expand_dims(neg_labels,axis=1)
		neg_labels = neg_labels.astype(object)
		intersected_boxes = intersected_boxes.astype(object)
		intersected_boxes = np.concatenate((intersected_boxes,neg_labels),axis=1)
	return np.concatenate((annotated_boxes,intersected_boxes),axis=0)

def get_average_mask_size(mask):
	return np.sum(mask.flatten() > 0) / mask.max()

def renumber_masks(mask):
	unique_m = np.sort(np.unique(mask))
	assert(unique_m[0] == 0)
	if unique_m[-1] == len(unique_m) - 1:
		return mask
	for i in range(1,len(unique_m)):
		mask[mask == unique_m[i]] = i
	return mask

def remove_edges(mask,min_edge_size=0.5):
	avg_mask_size = get_average_mask_size(mask)
	exclude_i = {}
	for i in range(mask.shape[0]):
		for j in [0,-1]:
			val = mask[i,j]
			if val > 0 and val not in exclude_i:
				exclude_i[val] = True
				if np.sum(mask.flatten() == val) < min_edge_size * avg_mask_size:
					mask[mask == val] = 0
	for i in range(mask.shape[1]):
		for j in [0,-1]:
			val = mask[j,i]
			if val > 0 and val not in exclude_i:
				exclude_i[val] = True
				if np.sum(mask.flatten() == val) < min_edge_size * avg_mask_size:
					mask[mask == val] = 0
	mask = renumber_masks(mask)
	return mask

def remove_smalls(mask,avg_fraction_thresh = 0.5):
	avg_mask_size = get_average_mask_size(mask)
	for i in range(1,mask.max() + 1):
		area = np.sum(mask == i)
		if area < avg_fraction_thresh * avg_mask_size:
			mask[mask == i] = 0
	mask = renumber_masks(mask)
	return mask

def masks_to_boxes(masks):
	if len(masks.shape) == 2:
		num_masks = len(np.unique(masks)) - 1
	elif len(masks.shape) == 3:
		num_masks = masks.shape[2]
	else:
		raise Exception("Invalid mask length: %d" % len(masks.shape))
	boxes = np.zeros((num_masks,4),dtype=int)
	for i in range(num_masks):
		if len(masks.shape) == 2:
			y_bar = np.max(masks == i+1,axis=0)
			x_bar = np.max(masks == i+1,axis=1)
		elif len(masks.shape) == 3:
			m = np.squeeze(masks[:,:,i])
			y_bar = np.max(m,axis=0)
			x_bar = np.max(m,axis=1)
		x_min = np.argmax(x_bar)
		y_min = np.argmax(y_bar)
		x_max = len(x_bar) - np.argmax(x_bar[::-1]) - 1
		y_max = len(y_bar) - np.argmax(y_bar[::-1]) - 1
		boxes[i,0] = y_min
		boxes[i,1] = x_min
		boxes[i,2] = y_max - y_min
		boxes[i,3] = x_max - x_min
		if i == 1 and False:
			print(boxes[i,:])
	return boxes

def get_im(im):
	if isinstance(im,str) and os.path.isfile(im):
		return np.array(Image.open(im))
	elif torch.is_tensor(im):
		return im.numpy()
	elif isinstance(im,np.ndarray):
		return im
	else:
		raise Exception("Unsupported type %s" % str(type(im)))

def get_cell_segmentation_mask(imagename,model = None, modelname = "cellpose",gpu=True):
	if isinstance(imagename,str):
		save_mask_name = "%s_mask.npy" % os.path.splitext(imagename)[0]
		if os.path.isfile(save_mask_name):
			print("Loading mask %s" % save_mask_name)
			return np.load(save_mask_name)
	img = get_im(imagename)#img = io.imread(imagename)
	if model == None:
		if modelname == "cellpose":
			try:
				model = models.Cellpose(gpu=gpu, model_type='cyto')
			except:
				raise Exception("""Attempt to instantiate cellpose model 
					did not work. Please check cellpose dependency""")
	if modelname == "cellpose":
		try:
			masks, flows, styles, diams = model.eval(img, diameter=None,channels=[0,0])
		except RuntimeError:
			model = models.Cellpose(gpu=False, model_type='cyto')
			masks, flows, styles, diams = model.eval(img, diameter=None,channels=[0,0])
	elif modelname == "deepcell":
		model = CytoplasmSegmentation()
		masks = model.predict(img, image_mpp=0.65)
	if isinstance(imagename,str):
		print("Saving mask %s" % save_mask_name)
		np.save(save_mask_name,masks)
	return masks


def get_touch_matrix(unique_vals,mask):
	unique_vals = list(unique_vals)
	touch_matrix = np.zeros((len(unique_vals),len(unique_vals))).astype(bool)
	for i in range(touch_matrix.shape[0]-1):
		for j in range(touch_matrix.shape[1]-1):
			if i == j:
				touch_matrix[i,j] = True
				touch_matrix[i+1,j+1] = True
			if mask[i,j] > 0:
				u1 = unique_vals.index(mask[i,j])
				for k in [(i,j+1),(i+1,j),(i+1,j+1)]:
					if mask[k[0],k[1]] > 0:
						u2 = unique_vals.index(mask[k[0],k[1]])
						touch_matrix[u1,u2] = True
						touch_matrix[u2,u1] = True
	return touch_matrix

def get_all_combos(l,include=None):
	if include is not None:
		l2 = list(l.copy())
		l2.remove(include)
		l = np.array(l2)
	all_combinations = []
	for r in [0,1]:#range(len(l) + 1):
		combinations_object = itertools.combinations(l, r)
		combinations_list = [cc for cc in list(combinations_object)]
		all_combinations += combinations_list
	if include is not None:
		all_combinations = [list(l) + [include] for l in all_combinations]
	return all_combinations

def do_function(mask,mean_area=None):
	unique_vals = np.unique(mask)
	unique_vals_list = list(unique_vals)
	if 0 in unique_vals_list:
		unique_vals_list.remove(0)
	else:
		print("0 not in unique vals")
	unique_vals = np.array(unique_vals_list)
	if len(unique_vals) > 10:
		tm = get_touch_matrix(unique_vals_list,mask)
	else:
		tm = None
	island_mapping = {}
	for u in unique_vals:
		if tm is None:
			all_combo_w_u = get_all_combos(unique_vals,include=u)
		else:
			#print(tm)
			#print(tm[:,unique_vals_list.index(u)].astype(bool))
			#print(unique_vals)
			assert(len(unique_vals) == tm.shape[0])
			combolist = unique_vals[tm[:,unique_vals_list.index(u)]]
			#print(combolist)
			if len(combolist) == 1:
				all_combo_w_u = [[u]]
			else:
				all_combo_w_u = get_all_combos(combolist,include=u)
		print(all_combo_w_u)
		for c in all_combo_w_u:
			smoothness = get_smoothness(mask,c,mean_area=mean_area)
			if u not in island_mapping or island_mapping[u][1] > smoothness:
				island_mapping[u] = (c,smoothness)
	ims = [island_mapping[u][0] for u in island_mapping]
	print("len(ims) %d" % len(ims))
	ims_nodups = []
	lolx = {}
	for im in ims:
		i_sort = sorted(im)
		ss = str(i_sort)
		if ss not in lolx:
			lolx[ss] = True
			ims_nodups.append(i_sort)
	print(ims_nodups)
	return ims_nodups

def resolve_intersections(masks):
	mean_area = np.mean([np.sum(masks == i) for i in range(1,masks.max()+1)])
	labeled_array, num_features = ndimage.label(masks != 0, np.ones((3,3)))
	all_pairs = []
	for i in np.unique(labeled_array):
		mask_copy = masks.copy()
		mask_copy[labeled_array != i] = 0
		mask_copy = mask_copy[~np.all(mask_copy == 0,axis=1),:]
		mask_copy = mask_copy[:,~np.all(mask_copy == 0,axis=0)]
		print(mask_copy)
		pairs = do_function(mask_copy,mean_area=mean_area)
		all_pairs += pairs
	masks_3D = np.zeros((masks.shape[0],masks.shape[1],len(all_pairs))).astype(bool)
	print("masks_3D.shape: %s"%str(masks_3D.shape))
	print("all_pairs len %d" % len(all_pairs))
	for i in range(len(all_pairs)):
		pair = all_pairs[i]
		for p in pair:
			masks_3D[:,:,i] = np.logical_or(masks_3D[:,:,i],masks == p)
	print("masks_3D.shape: %s"%str(masks_3D.shape))
	return masks_3D

def get_cell_boxes_from_image(imagename,model=None,gpu=True,
	remove_edge_cells=True,return_model=False,verbose=False,
	do_resolve_intersections=False,do_remove_smalls = True,
	do_remove_unsmooth=False):
	#temp = "temp.npy"
	#if not os.path.isfile(temp) and False:
	if verbose:
		masks = get_cell_segmentation_mask(imagename,gpu=gpu,model=model)
	else:
		with suppress_stdout():
			masks = get_cell_segmentation_mask(imagename,gpu=gpu,model=model)
	if remove_edge_cells:
		max_m = masks.max()
		masks = remove_edges(masks,min_edge_size=np.inf)
		if verbose: print("Removing %d cells from edges" % (max_m - masks.max()))
	#	np.save(temp,masks)
	#else:
	#	masks = np.load(temp)
	if do_remove_smalls:
		masks = remove_smalls(masks)
	# This gave some errors
	#if do_remove_unsmooth:
	#	masks = remove_unsmooth(masks)
	if do_resolve_intersections:
		print("masks.shape: %s" % str(masks.shape))
		print("unique_vals len - 1: %d" % (len(np.unique(masks))-1))
		masks = resolve_intersections(masks)
	boxes = masks_to_boxes(masks)
	if return_model:
		return boxes,masks,model
	else:
		return boxes,masks

def get_image_array(imagename,boxes,output_size = (64,64),
	whole_im_autoencoder=None):
	image = get_im(imagename)#np.array(Image.open(imagename))
	if whole_im_autoencoder is not None:
		image = tile_image(image,whole_im_autoencoder)
	X = np.zeros((boxes.shape[0],output_size[0],output_size[1],3))
	for i in range(boxes.shape[0]):
		x = boxes[i,0]
		y = boxes[i,1]
		l = boxes[i,2]
		w = boxes[i,3]
		imslice = slice_and_augment(image,x,y,l,w)
		if not type(imslice) == Image.Image:
			imslice = Image.fromarray(imslice)
		imslice = imslice.resize(output_size)
		imarr = np.expand_dims(np.array(imslice),axis=0)
		X[i,:,:,:] = imarr
	return X

def get_predicted_labels(imagename,single_cell_pred_model_filepath='.',
	gpu=True,remove_edge_cells = True,verbose=True,single_cell_pred_model=None,
	lb=None,segmodel=None,just_segment = False,autoencoder=None,whole_im_autoencoder=None,
	use_get_verify = False,num_dropout_tests = 1):
	if not just_segment and single_cell_pred_model is None and lb is None:
		single_cell_pred_model,lb = \
			load_model_w_label_processor(output_folder = \
			single_cell_pred_model_filepath,get_lb=True)
	boxes,masks = get_cell_boxes_from_image(imagename,gpu=gpu,model=segmodel)
	if just_segment:
		return None,masks
	X = get_image_array(imagename,boxes,whole_im_autoencoder=whole_im_autoencoder)
	if autoencoder is not None:
		X = autoencoder(X)
	X = X - X.min()
	X = X / X.max()
	X = X * 255.0
	if num_dropout_tests == 1:
		Y = single_cell_pred_model.predict(X,batch_size=1)
	else:
		for layer in single_cell_pred_model.layers:
			if "dropout" not in layer.name.lower():
				layer.trainable = False
		Y_many = np.array(
				[single_cell_pred_model(X,training=True) for _ in range(num_dropout_tests)]
			)
		print(np.std(Y_many,axis=0))
		Y = np.mean(Y_many,axis=0)
	Ym = np.argmax(Y[:,:len(lb.classes_)],axis=1)
	Y_uncertain = np.zeros((Y.shape[0]))
	for i in range(Y.shape[0]):
		if np.all(Y[i,:] > 0.0):
			Y_uncertain[i] = 1
	#assert(len(np.unique(Ym)) == 2)
	assert(len(np.unique(masks)) == masks.max() + 1)
	assert(masks.max() == len(Ym))
	Ym = np.expand_dims(Ym,axis=1)
	Yl = lb.inverse_transform(Ym)
	Yl=np.expand_dims(Yl,axis=1)
	for i in range(len(Ym)):
		if Y_uncertain[i] == 1:
			Yl[i] = "uncertain"
			masks[masks == i + 1] = -1
		else:
			masks[masks == i + 1] = Ym[i] + 1
	if not use_get_verify:
		return np.concatenate((boxes,Yl,Y[:,:len(lb.classes_)]),axis=1),masks
	else:
		Y = get_verify(X,Y[:,:len(lb.classes_) + 8])
		return np.concatenate((boxes,Yl,Y),axis=1),masks

def read_in_csv(labels):
	if isinstance(labels,str) and os.path.isfile(labels):
		foo = []
		with open(labels,'r') as fileobj:
			csvreader = csv.reader(fileobj)
			for row in csvreader:
				foo.append(row)
		labels = foo
	return labels

def get_classification_percent(labels,class_labels,get_total=False):
	results = {}
	total = 0
	labels = read_in_csv(labels)
	assert isinstance(labels,list)
	for c in class_labels:
		results[c] = 0
	for row in labels:
		total += 1
		l = row[4]
		results[l] += 1
	#if results["ring"] > 20: results["ring"] = 0
	if total == 0:
		return results
	if not get_total:
		for r in results:
			results[r] = results[r] / total
	return results

def segment_folder(folderpath,modelpath='.',random_sample=None,
	output_folderpath = None,model=None,lb=None,segmodel=None,autoencoder=None,
	whole_im_autoencoder=None,skip_if_present=False,shuffle_order=False):
	if output_folderpath == None:
		output_folderpath = folderpath
	imagenames = glob.glob(os.path.join(folderpath,'*.jpg'))
	if random_sample is not None:
		assert(isinstance(random_sample,int) and random_sample > 0)
		imagenames = list(imagenames)
		if shuffle_order:shuffle(imagenames)
		imagenames = imagenames[:random_sample]
	if not os.path.isdir(output_folderpath):
		os.makedirs(output_folderpath)
	for imagename in imagenames:
		basename = os.path.splitext(os.path.basename(imagename))[0]
		npy_save = os.path.join(output_folderpath,'%s_predicted.npy'%basename)
		csv_save = os.path.join(output_folderpath,'%s_predicted.csv'%basename)
		if os.path.isfile(npy_save) or os.path.isfile(csv_save) \
			and skip_if_present:
			continue
		labels,masks = get_predicted_labels(imagename,
			gpu=True,
			single_cell_pred_model_filepath=modelpath,
			single_cell_pred_model=model,
			lb=lb,
			segmodel=None,
			autoencoder=autoencoder,
			whole_im_autoencoder=whole_im_autoencoder,
			use_get_verify = True,num_dropout_tests = 10)
		#np.save(npy_save,masks)
		try:
			np.savetxt(csv_save,labels,delimiter=",",fmt='%s')
		except:
			print("Couldn't save the labels")

def get_labelnames_set(label_csv):
	ll = read_in_csv(label_csv)
	foo = []
	for row in ll:
		foo.append(row[4])
	return set(foo)

def get_folder_classification(folderpath,get_total=False,get_everything = False):
	labelnames = glob.glob(os.path.join(folderpath,'*.csv'))
	class_labels = set([])
	for labelname in labelnames:
		class_labels = class_labels.union(get_labelnames_set(labelname))
	class_labels = sorted(list(class_labels))
	all_results = {}
	for labelname in labelnames:
		result = get_classification_percent(labelname,class_labels,
			get_total=get_total)
		if True:#result["ring"] < 16:
			for r in result:
				if r not in all_results:
					all_results[r] = []
				all_results[r].append(result[r])
	if get_total and get_everything:
		return all_results
	for r in all_results:
		all_results[r] = float(np.mean(all_results[r]))
	return all_results


def get_file_list_from_str(obj):
	assert(isinstance(obj, str))
	if os.path.isfile(obj):
		if is_image_file(obj):
			return [obj]
		else:
			return []
	elif os.path.isdir(obj):
		all_filename_list = []
		for root, dirs, files in os.walk(obj, topdown=False):
			for name in files:
				filename = os.path.join(root, name)
				if is_image_file(filename):
					all_filename_list.append(filename)
		return all_filename_list
	else:
		raise Exception("Invalid string input: %s" % obj)

def get_file_list_from_list(obj,allow_list_of_list=True):
	assert(isinstance(obj,list))
	if np.all([isinstance(_,list) for _ in obj]):
		if not allow_list_of_list:
			raise Exception("Cannot have nested lists")
		list_of_list = []
		for l in obj:
			list_of_list = list_of_list + get_file_list(l,
				allow_list_of_list = False)
		return list_of_list
	elif np.all([isinstance(_,str) for _ in obj]):
		list_of_str = []
		for l in obj:
			list_of_str = list_of_str + get_file_list_from_str(l)
		return list_of_str
	else:
		raise Exception("""Inputs must be strings, lists of lists,
			or lists of strings""")

def get_file_list(obj,allow_list_of_list=True):
	if isinstance(obj,str):
		obj = get_file_list_from_str(obj)
	elif isinstance(obj,list):
		obj = get_file_list_from_list(obj)
	else:
		raise Exception("Invalid path input")
	assert(isinstance(obj,list))
	assert(np.all([isinstance(_,str) for _ in obj]))
	if np.all([len(_) == 0 for _ in obj]):
		raise Exception("No valid files found")
	elif np.any([len(_) == 0 for _ in obj]):
		raise Exception("One without valid files")
	return obj

if __name__ == '__main__':
	imagename = sys.argv[1]
	output_folder = os.path.join('.','image_segment_examples')
	just_segment = False
	if not os.path.isdir(os.path.join('.','models')):
		just_segment = True
	if not os.path.isdir(output_folder):
		os.mkdir(output_folder)
	foldername = "%s_%s" % (os.path.basename(os.path.dirname(imagename)),
		os.path.splitext(os.path.basename(imagename))[0])
	image_output_folder = os.path.join(output_folder,foldername)
	if not os.path.isdir(image_output_folder):
		os.mkdir(image_output_folder)
	labels,masks = get_predicted_labels(imagename,gpu=True,just_segment=just_segment)
	
	np.save(os.path.join(image_output_folder,'masks.npy'),masks)
	if not just_segment:
		np.save(os.path.join(image_output_folder,'masks_sum.npy'),masks)
		np.savetxt(os.path.join(image_output_folder,'labels.csv'),labels,
			delimiter=",",fmt='%s')
	copyfile(imagename,os.path.join(image_output_folder,os.path.basename(imagename)))
	print(imagename)
