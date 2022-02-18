import os

import imageio

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')


def write_to_gif(gif_path, images, fps=30):
    with imageio.get_writer(gif_path, mode='I') as writer:
        for image in images:
            writer.append_data(image)


def get_frame_index(path):
    file_name = os.path.basename(path)
    return int(file_name.split('.')[0].rsplit('_', 1)[-1])


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 0
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    # initialize logger
    wandb_run = wandb.init(project='CycleGAN-and-pix2pix', name=opt.name, config=opt) if not wandb.run else wandb.run
    wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name,
                           '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    frame_data = []
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        frame_data.append([img_path[0], visuals['fake_B'].cpu()[0].permute(1, 2, 0).numpy(),
                           visuals['real_B'].cpu()[0].permute(1, 2, 0).numpy()])
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
    frame_data = sorted(frame_data, key=lambda x: int(get_frame_index(frame_data[0])))
    save_images(webpage, frame_data, opt.results_dir, opt.name, opt.phase)
