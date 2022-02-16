import glob
import os
from collections import defaultdict

import torch

from data.base_dataset import BaseDataset, get_transform
from PIL import Image
import random


class VideoDataset(BaseDataset):
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

        self.A_paths = self._load_video_path(self.dir_A)  # load images from '/path/to/data/trainA'
        self.B_paths = self._load_video_path(self.dir_B)  # load images from '/path/to/data/trainB'
        self.A_videos = list(self.A_paths.keys())
        self.B_videos = list(self.B_paths.keys())
        self.A_size = len(self.A_videos)  # get the size of dataset A
        self.B_size = len(self.B_videos)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.transform_A = get_transform(self.opt, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, grayscale=(output_nc == 1))
        self.video_length = 32

    def _load_video_path(self, path):
        self.video_path = defaultdict(list)
        for video_name in glob.glob(os.path.join(path, '*')):
            for video_file in glob.glob(os.path.join(path, video_name, '*')):
                self.video_path[video_name].append(video_file)
            self.video_path[video_name] = sorted(self.video_path[video_name])
        return self.video_path

    def _load_video(self, video_path):
        images = []
        for image_path in video_path:
            frame = Image.open(image_path).convert('RGB')
            frame = self.transform_A(frame)
            images.append(frame)
        return self._crop_and_pad_video(torch.stack(images))

    def _crop_and_pad_video(self, video):
        # crop and pad video to the same size
        # video: (3, T, H, W) with T=16 frames
        # return: (3, T, H, W) with T=16 frames
        T, C, H, W = video.size()
        if T >= self.video_length:
            video = video[:self.video_length, ...]
        elif T < self.video_length:
            repeat_num = self.video_length // T + 1
            video = video.repeat(repeat_num, 1, 1, 1)[:self.video_length, ...]
        return video.permute(1, 0, 2, 3)

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
        A_video = self.A_paths[self.A_videos[index % self.A_size]]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_video = self.B_paths[self.B_videos[index_B]]
        A = self._load_video(A_video)
        B = self._load_video(B_video)
        return {'A': A, 'B': B, 'A_paths': A_video, 'B_paths': B_video}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)


if __name__ == '__main__':
    opt = torch.load("opt.pth")
    dataset = VideoDataset(opt)
    for i in dataset:
        print(i['A'].shape)
