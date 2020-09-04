import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from module.udagan import Augmenter_mnist, Discriminator_mnist
from module.dataloader import get_data
from module.utils import *
from matplotlib import gridspec

fontsize = 6
path = './results/mnist/'
file = 'model_bs_64_dn_12_dz_10_lambda_0' #'model_bs_256_dn_12_dz_10'
load_file = path + file
# Load the checkpoint file
model = torch.load(load_file)

# Set the device to run on: GPU or CPU
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
# Get the 'parameters' dictionary from the loaded file
parameters = model['parameters']
n_samp = parameters['n_arm']

# Initialise the network
netA = Augmenter_mnist(noise_dim=parameters['num_n'],
                 latent_dim=parameters['num_z']).to(device)
# netD = Discriminator().to(device)
# Load the trained augmenter weights
netA.load_state_dict(model['netA'])
# netD.load_state_dict(state_dict['netD'])

dataloader = get_data(dataset='MNIST',
                      batch_size=parameters['batch_size'],
                      file_path='',
                      n_feature=[])

with torch.no_grad():
    for i, (data, _) in enumerate(dataloader, 0):
        # Get batch size
        if i < 50:
            b_size = parameters['batch_size']
            # Transfer data tensor to GPU/CPU (device)
            real_data = data.to(device)
            augmented_img = []
            err = []

            for arm in range(parameters['n_arm']):
                noise = torch.randn(b_size, parameters['num_n'], device=device)
                noise += 0.1 * torch.sign(noise)
                # zeros = torch.zeros(b_size, parameters['num_n'], 1, 1,
                #                     device=device)
                _, gen_data = netA(real_data, noise)
                augmented_img.append(gen_data.detach().cpu())

            ns = 15
            fig = plt.figure(figsize=(4, 3))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, parameters['n_arm']])
            ax = plt.subplot(gs[0])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Original Data')
            plt.imshow(np.transpose(vutils.make_grid(
                real_data[:ns, :, :, :].detach().cpu(), nrow=1,
                padding=2, normalize=True), (1, 2, 0)), aspect='auto')
            concat_img = torch.zeros((1, real_data.size(2),
                                      real_data.size(-1)))
            for n_smp in range(ns):
                for arm in range(parameters['n_arm']):
                    concat_img = torch.cat((concat_img, augmented_img[arm][
                                              n_smp, :, :, :]), dim=0)
            concat_img = concat_img.unsqueeze(1)
            ax = plt.subplot(gs[1])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel('Augmented Data')
            plt.imshow(
                np.transpose(vutils.make_grid(concat_img[1:, :, :, :],
                        nrow=parameters['n_arm'], padding=2, normalize=True),
                             (1, 2, 0)), aspect='auto')
            plt.subplots_adjust(wspace=0.1)
            plt.tight_layout()
            plt.savefig(path + 'test_sample_' + str(i) + '.png', dpi=300)
            plt.close('all')

        else:
            break
