import os

import cv2
import torch
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
from torchvision import transforms
import numpy as np

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random


class MultiChannelDataset(BaseDataset):
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

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))

    def _get_next_frame_path(self, frame_path):
        """
        Get the next frame path from the given frame path.
        """
        # remove file extension
        frame_path, file_extension = frame_path.rsplit('.', 1)
        frame_path_list = frame_path.rsplit('_', 1)
        frame_idx = int(frame_path_list[-1])
        next_frame_idx = frame_idx + 1
        next_frame_path = '_'.join(frame_path_list[:-1]) + '_' + "{:04d}".format(next_frame_idx) + '.' + file_extension
        if not os.path.exists(next_frame_path):
            next_frame_path = '_'.join(frame_path_list[:-1]) + '_' + "{:04d}".format(next_frame_idx - 2) + '.' + file_extension

        assert os.path.exists(next_frame_path), 'The next frame path {} does not exist'.format(next_frame_path)
        return next_frame_path

    def convert_channel(self, frame):
        frame = np.array(frame)
        frame = cv2.resize(frame, (256, 256), interpolation=cv2.INTER_NEAREST)
        frame = torch.Tensor(frame)
        frame = torch.Tensor(frame)
        frame = one_hot(frame.to(torch.long))
        frame = frame.permute(2, 0, 1).to(torch.float32)
        return frame

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
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path)
        A_next = Image.open(self._get_next_frame_path(A_path))

        A = self.convert_channel(A_img)
        A_next = self.convert_channel(A_next)

        B_img = Image.open(B_path).convert('RGB')
        B = self.transform_B(B_img)

        return {'A': A, 'A_next': A_next, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


if __name__ == '__main__':
    opt = torch.load("../opt.pth")
    dataset = MultiChannelDataset(opt)
    for i in dataset:
        image = i['A_next']
        # visualize matplotlib
        for j in range(image.shape[-1]):
            cv2.imshow("image", np.uint8(image[:, :, j].numpy()*255.0))
            cv2.waitKey(0)
        break