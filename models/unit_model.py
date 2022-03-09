import torch
import yaml
from torch import nn

from .base_model import BaseModel
from .networks import MsImageDis, VAEGen


class UNITModel(BaseModel):
    """
    This class implements the UNIT model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    MUNIT paper: https://arxiv.org/pdf/1804.04732.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--munit_config_path', type=str, default='munit_config/config.yaml',
                                help='Path to the config file.')
        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ["gen_recon_x_a", "gen_recon_x_b", "gen_recon_kl_a",
                           "gen_recon_kl_b", "gen_cyc_x_a", "gen_cyc_x_b",
                           "gen_recon_kl_cyc_aba", "gen_recon_kl_cyc_bab",
                           "gen_adv_a", "gen_adv_b", "gen_total",
                           "dis_a", "dis_b", "dis_total"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', "real_A_recon"]
        visual_names_B = ['real_B', 'fake_A', "real_B_recon"]

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.configs = yaml.load(open(opt.munit_config_path, 'r'))
        if len(self.gpu_ids) > 1:
            raise NotImplementedError("Multi-GPU is not supported yet.")
        self.netG_A = VAEGen(opt.input_nc, self.configs['gen']).to(self.gpu_ids[0])  # auto-encoder for domain a
        self.netG_B = VAEGen(opt.input_nc, self.configs['gen']).to(self.gpu_ids[0])  # auto-encoder for domain b
        self.netD_A = MsImageDis(opt.input_nc, self.configs['dis']).to(self.gpu_ids[0])  # discriminator for domain a
        self.netD_B = MsImageDis(opt.input_nc, self.configs['dis']).to(self.gpu_ids[0])  # discriminator for domain b
        self.instancenorm = nn.InstanceNorm2d(512, affine=False).to(self.gpu_ids[0])
        self.style_dim = self.configs['gen']['style_dim']

        # fix the noise used in sampling
        display_size = int(self.configs['display_size'])

        # Setup the optimizers
        # TODO
        # Network weight initialization
        # self.apply(weights_init(self.configs['init']))
        # self.netD_A.apply(weights_init('gaussian'))
        # self.netD_B.apply(weights_init('gaussian'))

        # Load VGG model if needed
        # if 'vgg_w' in hyperparameters.keys() and hyperparameters['vgg_w'] > 0:
        #     self.vgg = load_vgg16(hyperparameters['vgg_model_path'] + '/models')
        #     self.vgg.eval()
        #     for param in self.vgg.parameters():
        #         param.requires_grad = False

        # self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        # self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
        #                                 not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        #
        # if self.isTrain:  # define discriminators
        #     self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
        #                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        #     self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
        #                                     opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
        beta1, beta2 = opt.beta1, 0.9999
        if self.isTrain:
            # define loss functions
            dis_params = list(self.netD_A.parameters()) + list(self.netD_B.parameters())
            gen_params = list(self.netG_A.parameters()) + list(self.netG_B.parameters())
            self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad],
                                            lr=opt.lr, betas=(beta1, beta2))
            self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad],
                                            lr=opt.lr, betas=(beta1, beta2))
            self.optimizers = [self.gen_opt, self.dis_opt]

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def recon_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # encode
        self.h_a, self.n_a = self.netG_A.encode(self.real_A)
        self.h_b, self.n_b = self.netG_B.encode(self.real_B)
        # decode (within domain)
        self.real_A_recon = self.netG_A.decode(self.h_a + self.n_a)
        self.real_B_recon = self.netG_B.decode(self.h_b + self.n_b)
        # decode (cross domain)
        self.fake_A = self.netG_A.decode(self.h_b + self.n_b)
        self.fake_B = self.netG_B.decode(self.h_a + self.n_a)
        # encode again
        self.h_b_recon, self.n_b_recon = self.netG_A.encode(self.fake_A)
        self.h_a_recon, self.n_a_recon = self.netG_B.encode(self.fake_B)
        # decode again (if needed)
        self.x_aba = self.netG_A.decode(self.h_a_recon + self.n_a_recon) if self.configs['recon_x_cyc_w'] > 0 else None
        self.x_bab = self.netG_B.decode(self.h_b_recon + self.n_b_recon) if self.configs['recon_x_cyc_w'] > 0 else None

    def backward_G(self):
        self.loss_gen_recon_x_a = self.recon_criterion(self.real_A_recon, self.real_A)
        self.loss_gen_recon_x_b = self.recon_criterion(self.real_B_recon, self.real_B)
        self.loss_gen_recon_kl_a = self.__compute_kl(self.h_a)
        self.loss_gen_recon_kl_b = self.__compute_kl(self.h_b)
        self.loss_gen_cyc_x_a = self.recon_criterion(self.x_aba, self.real_A) if self.x_aba is not None else 0
        self.loss_gen_cyc_x_b = self.recon_criterion(self.x_bab, self.real_B) if self.x_bab is not None else 0
        self.loss_gen_recon_kl_cyc_aba = self.__compute_kl(self.h_a_recon)
        self.loss_gen_recon_kl_cyc_bab = self.__compute_kl(self.h_b_recon)
        # GAN loss
        self.loss_gen_adv_a = self.netD_A.calc_gen_loss(self.fake_A)
        self.loss_gen_adv_b = self.netD_B.calc_gen_loss(self.fake_B)
        # domain-invariant perceptual loss
        # self.loss_gen_vgg_a = self.compute_vgg_loss(self.vgg, self.x_ba, self.x_b) if self.configs['vgg_w'] > 0 else 0
        # self.loss_gen_vgg_b = self.compute_vgg_loss(self.vgg, self.x_ab, self.x_a) if self.configs['vgg_w'] > 0 else 0
        self.loss_gen_total = self.configs['gan_w'] * self.loss_gen_adv_a + \
                              self.configs['gan_w'] * self.loss_gen_adv_b + \
                              self.configs['recon_x_w'] * self.loss_gen_recon_x_a + \
                              self.configs['recon_kl_w'] * self.loss_gen_recon_kl_a + \
                              self.configs['recon_x_w'] * self.loss_gen_recon_x_b + \
                              self.configs['recon_kl_w'] * self.loss_gen_recon_kl_b + \
                              self.configs['recon_x_cyc_w'] * self.loss_gen_cyc_x_a + \
                              self.configs['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_aba + \
                              self.configs['recon_x_cyc_w'] * self.loss_gen_cyc_x_b + \
                              self.configs['recon_kl_cyc_w'] * self.loss_gen_recon_kl_cyc_bab
        # total loss

        self.loss_gen_total.backward()

    def backward_D(self):
        self.dis_opt.zero_grad()
        # encode
        h_a, n_a = self.netG_A.encode(self.real_A)
        h_b, n_b = self.netG_B.encode(self.real_B)
        # decode (cross domain)
        x_ba = self.netG_A.decode(h_b + n_b)
        x_ab = self.netG_B.decode(h_a + n_a)
        # D loss
        self.loss_dis_a = self.netD_A.calc_dis_loss(x_ba.detach(), self.real_A)
        self.loss_dis_b = self.netD_B.calc_dis_loss(x_ab.detach(), self.real_B)
        self.loss_dis_total = self.configs['gan_w'] * self.loss_dis_a + self.configs['gan_w'] * self.loss_dis_b
        self.loss_dis_total.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        self.backward_G()  # calculate gradients for G
        self.backward_D()  # calculate gradients for D
        self.gen_opt.step()
        self.dis_opt.step()

    def __compute_kl(self, mu):
        mu_2 = torch.pow(mu, 2)
        encoding_loss = torch.mean(mu_2)
        return encoding_loss
