Cell Data Loader
================

Cell Data Loader is a simple AI support tool in Python that can take in images of cells (or other image types) and output them with minimal effort to formats that can be read by Pytorch (Tensor) or Tensorflow (Numpy) format. With Cell Data Loader, users have the option to output their cell images as whole images, sliced images, or, with the support of [CellPose](https://github.com/MouseLand/cellpose), segment their images by cell and output those individually.

It can also be used for normal computer vision research, which is why CellPose is not a strict dependency.

Quick Start
-----------

To install Cell Data Loader, simply type into a standard UNIX terminal

    pip install cell-data-loader

To use a quick Jupyter Notebook example, navigate to [example/CellDataLoaderPlayground.ipynb](https://github.com/mleming/CellDataLoader/blob/main/example/CellDataLoaderPlayground.ipynb), download it, and run it in [Jupyter Notebook](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html). On a Mac, that may be run with the following command:

	python3 -m jupyterlab ~/Downloads/CellDataLoaderPlayground.ipynb


Python
------

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

Boxes
-----

In cases where you need to cut out individual cells from an image and have the coordinates file, cell_data_loader.py accepts an argument, cell_box_filelist, which is a list of files corresponding to the inputs that mark out the coordinates of labels on the cells. The format of the csv is as follows:

| X  | Y   | W  | H | Label |
| -  | -   | -  | - | ----- |
| 14 | 13  | 5  | 6 | 0     |
| 20 | 25  | 15 | 5 | 1     |


~~~python
dataloader = CellDataloader('/path/to/file.svs',cell_box_filelist=['/path/to/boxfile.csv'])
~~~

Arguments
---------

Additional arguments taken by Cell Data Loader include

~~~python

imfolder = '/path/to/folder'

dataloader = CellDataloader(imfolder,
			dim = (64,64),
			batch_size = 64,
			dtype = "torch", # Can also be "numpy"
			label_regex = None,
			verbose = True,
			segment_image = "whole", # "whole" outputs the whole image, resized
				# to dim; "sliced" cuts the image checkerboard pattern into
				# dim-shaped outputs, so it's suitable for large images; "cell"
				# segments cells from the image using CellPose, though it throws
				# an error if CellPose is not installed properly. CellPose is
				# not included by default in the dependencies and needs to be
				# installed separately by the user.
			n_channels = 3, # Detected in first image by default; re-samples all
				# images to force this number of channels
			augment_image = True, # Augments the output image in the standard
				# ways -- rotation, color jiggling, etc.
			label_balance = True, # Outputs proportional amounts of each label
				# in the dataset
			gpu_ids = None, # GPUs that the outputs are read to, if present.
			channels_first = True # Places channels either first, before the
				# batch dimension, or last
			)
~~~


Dependencies
------------

Note that the strict dependencies are automatically downloaded just with

	pip install cell-data-loader

However, to get support with cell-segmentation-specific images (i.e., segment="cell"), [CellPose](https://github.com/MouseLand/cellpose) needs to be installed. GPU integration with CellPose would also need to be handled separately.

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

