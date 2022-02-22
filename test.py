import IPython
import imageio
import numpy as np

from data import create_dataset
from models import create_model
from options.test_options import TestOptions

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
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers

    frame_data = []
    for i, data in enumerate(dataset):
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        model.forward()  # run inference

        if i > 100:
            break
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()  # get image paths
        frame_data.append([img_path[0],
                           np.hstack([visuals['fake_B'].cpu()[0].permute(1, 2, 0).detach().numpy(),
                                      visuals['real_A'].cpu()[0].permute(1, 2, 0).detach().numpy()])
                           ])
        break
    for k in visuals.keys():
        imageio.imsave(f'{k}.png', visuals[k].cpu()[0].permute(1, 2, 0).detach().numpy())
    IPython.embed()

    frame_data = np.array([x[1] for x in frame_data])
    # convert to channels first to fit wandb logger
    frame_data = frame_data.transpose(0, 3, 1, 2)
    video_results = frame_data
    video_results = (video_results - video_results.min()) / (video_results.max() - video_results.min())
    video_results = np.uint8(video_results * 255)

    wandb_run.log({
        "sequence A to B": wandb.Video(video_results, fps=10, format="gif")
    })
