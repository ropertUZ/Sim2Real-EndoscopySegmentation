import os
from data.base_dataset import BaseDataset, get_transform, get_params
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import torch

class UnalignedDepthDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        if opt.phase == 'train':
            self.dir_B_depth = os.path.join(opt.dataroot,opt.phase + 'B_depth') # Load GT depth map

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        if opt.phase == 'train':
            self.B_depth_paths = sorted(make_dataset(self.dir_B_depth, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        
        
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        if self.opt.phase == 'train':
            B_depth_path = self.B_depth_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        if self.opt.phase == 'train':
            B_depth_img = Image.open(B_depth_path).convert('RGB')
        # apply image transformation
        params_A = get_params(self.opt, A_img.size)
        params_B = get_params(self.opt, B_img.size)
        self.transform_A = get_transform(self.opt, params_A, grayscale=(self.input_nc == 1))
        self.transform_B = get_transform(self.opt, params_B, grayscale=(self.output_nc == 1))
        if self.opt.phase == 'train':
            self.transform_B_depth = get_transform(self.opt, params_B, grayscale=True, binary=False)#True)
            B_depth_img = self.transform_B_depth(B_depth_img)
        

        A = self.transform_A(A_img)
        B = self.transform_B(B_img)
        if self.opt.phase == 'train':
            return {'A': A, 'B': B, 'B_depth' : B_depth_img, 'A_paths': A_path, 'B_paths': B_path}
        else:
            return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
