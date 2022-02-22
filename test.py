import os

import imageio
import numpy as np
import torch

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import get_frame_index, write_to_gif
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name="test") if not wandb.run else wandb.run
    wandb_run._label(repo='CycleGAN-and-pix2pix')
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    for i in dataset:
        print(i["A_paths"])
    # model = create_model(opt)  # create a model given opt.model and other options
    # model.setup(opt)  # regular setup: load and print networks; create schedulers
    #
    #
    #
    # if opt.eval:
    #     model.eval()
    # frame_data = []
    # for i, data in enumerate(dataset):
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()  # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()  # get image paths
    #     frame_data.append([img_path[0], visuals['fake_B'].cpu()[0].permute(1, 2, 0).numpy(),
    #                        visuals['real_A'].cpu()[0].permute(1, 2, 0).numpy()])
    #     if i % 5 == 0:
    #         print('processing (%04d)-th image... %s' % (i, img_path))

    # frame_data = sorted(frame_data, key=lambda x: int(get_frame_index(x[0])))
    # frame_data = [np.hstack([x[1], x[2]]) for x in frame_data]
    # frame_data = np.array(frame_data)
    # frame_data = frame_data.transpose(0, 3, 1, 2)
    # torch.save(frame_data, "frame.pth")
    frame_data = torch.load("frame.pth")
    # scale frames to 0-1
    frame_data = (frame_data - frame_data.min()) / (frame_data.max() - frame_data.min())
    frame_data = np.uint8(frame_data * 255)
    wandb_run.log({
        "sequence A to B": wandb.Video(frame_data, fps=10, format="gif")
    })
