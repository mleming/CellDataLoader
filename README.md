Cell Data Loader
================

Cell Data Loader is a simple AI support tool in Python that can take in images of cells (or other image types) and output them with minimal effort to formats that can be read by Pytorch (Tensor) or Tensorflow (Numpy) format. With Cell Data Loader, users have the option to output their cell images as whole images, sliced images, or, with the support of [CellPose](https://github.com/MouseLand/cellpose), segment their images by cell and output those individually.

To install Cell Data Loader, simply type into a standard UNIX terminal

    pip install cell_data_loader

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
imfiles = ['/path/to/CANCER_image1.png','/path/to/CANCER_image2.png','/path/to/CANCER_image3.png','/path/to/HEALTHY_image1.png','/path/to/HEALTHY_image2.png','/path/to/HEALTHY_image3.png']

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
			dtype = "numpy",
			n_channels = 3, #This is detected in the first read image by default, if not provided; it re-samples all images to force this number of channels
			)
~~~


Dependencies
------------

Strict dependencies:

> scipy
> numpy
> pytorch

Soft dependencies:
> CellPose # For cell segmentation support
> slideio # For reading .SVS and .CZI files


Note that some of the dependencies are not strict and vary depending on usage. Numpy and 
