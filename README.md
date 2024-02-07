Cell Data Loader
================

Cell Data Loader is a simple AI support tool in Python that can take in images of cells (or other image types) and output them with minimal effort to formats that can be read by Pytorch (Tensor) or Tensorflow (Numpy) format. With Cell Data Loader, users have the option to output their cell images as whole images, sliced images, or, with the support of [CellPose](https://github.com/MouseLand/cellpose), segment their images by cell and output those individually.

It can also be used for normal computer vision research, which is why CellPose is not a strict dependency.

To install Cell Data Loader, simply type into a standard UNIX terminal

    pip install cell-data-loader

The simplest way to use Cell Data Loader is to instantiate a dataloader as such:

~~~python
from cell_data_loader import CellDataloader

imfolder = '/path/to/my/images'

dataloader = CellDataloader(imfolder)

for image in dataloader:
	...
~~~

And viola!

Lists of files are also supported:

~~~python

imfiles = ['/path/to/image1.png','/path/to/image2.png','/path/to/image3.png']

dataloader = CellDataloader(imfiles)

for image in dataloader:
	...
~~~

Labels
------

Cell Data Loader has a few ways to support image labels. The simplest is whole images that are located in different folders, with each folder representing a label. This can be supported via the following:

~~~python
imfolder1 = '/path/to/my/images'
imfolder2 = '/path/to/your/images'

dataloader = CellDataloader(imfolder1,imfolder2)

for label,image in dataloader:
	...
~~~

Alternatively, if you have one folder or file list with images that have different naming conventions, a regex match is supported:

~~~python
imfiles = ['/path/to/CANCER_image1.png',
			'/path/to/CANCER_image2.png',
			'/path/to/CANCER_image3.png',
			'/path/to/HEALTHY_image1.png',
			'/path/to/HEALTHY_image2.png',
			'/path/to/HEALTHY_image3.png']

dataloader = CellDataloader(imfiles,label_regex = ["CANCER","HEALTHY"])
for label,image in dataloader:
	...
~~~


Arguments
---------

Additional arguments taken by Cell Data Loader include

~~~python

imfolder = '/path/to/folder'

dataloader = CellDataloader(imfolder,
			dim = (64,64),
			batch_size = 32,
			dtype = "numpy", # Can also be "torch"
			label_regex = None,
			n_channels = 3, # Detected in first image by default; re-samples all images to force this number of channels
			match_labels = False, # If true, outputs proportional amounts of each label in the dataset
			gpu_ids = None # GPUs that the outputs are read to, if present.
			)
~~~


Dependencies
------------

Strict dependencies:

	numpy
	torch
	torchvision
	opencv-python>=4.5.4
	slideio==2.4.1
	scipy
	scikit-image
	pillow

Soft dependencies:

	CellPose # For cell segmentation support
	Tensorflow

Note that some of the dependencies are not strict and vary depending on usage. Numpy is a hard requirement, but Tensorflow is not if the user only uses the Torch capabilities. If the user attempts to load cell images in "cell" mode without a working Cellpose installation, CellDataLoader will throw an error. Cellpose needs be be installed separately to use "cell" mode:

	pip install cellpose

And GPU integration for Cellpose would need to be handled separately.
