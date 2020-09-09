import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from module.udagan import Generator_facs
from module.dataloader import get_data
import os
import pandas as pd
from module.scRNA_AE import *
import seaborn as sns
from matplotlib import gridspec

fontsize = 6
path = os.path.abspath(os.path.join(os.getcwd(), '..'))
path += '/results/'
file_gen = 'generator/scRNA/modelG_bs_1000_dn_10_dz_1000'
file_ae = 'ae/run_0_s_2_drop_50.0_fc_dim_100_e_10000_batchS_1000_lr_0' \
          '.001_mom_0.05/' + 'model/singleAE_model_2020-08-29-23-55-19'
file_marker_genes = '/home/yeganeh/data/celltypes/facs-10x-ss-highexp-gene' \
                    '-list.csv'
load_file_gen = path + file_gen
load_file_ae = path + file_ae
# Load the checkpoint file
model_gen = torch.load(load_file_gen)

# Set the device to run on: GPU or CPU
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
# Get the 'parameters' dictionary from the loaded file
parameters = model_gen['parameters']
n_samp = parameters['n_smp']

# Initialise the networks
netG = Generator_facs(latent_dim=parameters['num_z'],
                      input_dim=parameters['n_features']).to(device)
netG.load_state_dict(model_gen['netG'])

netAE = singleAE(input_dim=parameters['n_features'],
                fc_dim=100,
                latent_dim=2,
                p_drop=0.).to(device)
netAE.load_state_dict(torch.load(load_file_ae))

dataloader, dataset = get_data(dataset=parameters['dataset'],
                              batch_size=parameters['batch_size'],
                              file=parameters['dataset_file'],
                              n_feature=parameters['n_features'],
                              training=False,
                              remove_nonneuron=parameters['remove_nonneuron'],
                              remove_CR_Meis2=parameters['remove_CR_Meis2'])

data_real_low = np.zeros((len(dataloader.dataset), 2))
data_gen_low = np.zeros((len(dataloader.dataset), 2))
data_bin_low = np.zeros((len(dataloader.dataset), 2))
data_generate = np.zeros((len(dataloader.dataset), parameters[
    'n_features']))
data_origin = np.zeros((len(dataloader.dataset), parameters['n_features']))
samp_clr = ["" for x in range(len(dataloader.dataset))]
cluster = ["" for x in range(len(dataloader.dataset))]

with torch.no_grad():
    for i, (data, data_bin, labels) in enumerate(dataloader):
        # Get batch size
        b_size = parameters['batch_size']
        # Generate augmented samples
        real_data = data.to(device)
        real_data_bin = data_bin.to(device)
        _, gen_data = netG(real_data_bin)
        # Get the lowD representation
        _, z_real = netAE(real_data)
        _, z_gen = netAE(gen_data)
        _, z_bin = netAE(real_data_bin)
        l = [int(lab) for lab in labels.numpy()]
        samp_clr[i * b_size:min((i + 1) * b_size,
                    len(dataloader.dataset))] = dataset['cluster_color'][l]
        cluster[i * b_size:min((i + 1) * b_size,
                    len(dataloader.dataset))] = dataset['cluster'][l]
        data_real_low[i * b_size:min((i + 1) * b_size,
                            len(dataloader.dataset)), :] = \
            z_real.cpu().detach().numpy()
        data_gen_low[i * b_size:min((i + 1) * b_size,
                            len(dataloader.dataset)), :] = \
            z_gen.cpu().detach().numpy()
        data_bin_low[i * b_size:min((i + 1) * b_size,
                            len(dataloader.dataset)), :] = \
            z_bin.cpu().detach().numpy()
        data_generate[i * b_size:min((i + 1) * b_size,
                            len(dataloader.dataset)), :] = \
            gen_data.cpu().detach().numpy()
        data_origin[i * b_size:min((i + 1) * b_size,
                                     len(dataloader.dataset)), :] = \
            real_data.cpu().detach().numpy()

axis_lim = 2.7
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
ax1.scatter(data_real_low[:, 0], data_real_low[:, 1], color=samp_clr,
            s=1.5, alpha=0.8)
ax1.set_xlabel(r'$s_1$', fontsize=18)
ax1.set_ylabel(r'$s_2$', fontsize=18)
ax1.set_xlim(-axis_lim, axis_lim)
ax1.set_ylim(-axis_lim, axis_lim)
ax1.set_title('Original Samples', fontsize=16)

ax2.scatter(data_gen_low[:, 0], data_gen_low[:, 1], color=samp_clr,
            s=1.5, alpha=0.8)
ax2.set_xlabel(r'$s_1$', fontsize=18)
ax2.set_ylabel(r'$s_2$', fontsize=18)
ax2.set_title('Generated Samples', fontsize=16)
ax2.set_xlim(-axis_lim, axis_lim)
ax2.set_ylim(-axis_lim, axis_lim)

# ax3.scatter(data_bin_low[:, 0], data_bin_low[:, 1], color=samp_clr,
#             s=1.5, alpha=0.8)
# ax3.set_xlabel(r'$s_1$', fontsize=18)
# ax3.set_ylabel(r'$s_2$', fontsize=18)
# ax3.set_title('Binary Samples', fontsize=16)
# ax3.set_xlim(-axis_lim, axis_lim)
# ax3.set_ylim(-axis_lim, axis_lim)
plt.savefig(path + 'generator/scRNA/lowD_feature_space.png')

# Plot per type
unique_cluster = np.unique(dataset['cluster'])
cluster = np.array(cluster)
count = 0
for type in unique_cluster:
    count +=1
    indx = np.where(cluster == type)[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(data_real_low[indx, 0], data_real_low[indx, 1], color='blue',
                s=10, alpha=0.6, marker='o', label='Original Sample')
    ax.scatter(data_gen_low[indx, 0], data_gen_low[indx, 1], color='red',
                s=8, alpha=0.6, marker='^', label='Generated Sample')
    ax.scatter(data_bin_low[indx, 0], data_bin_low[indx, 1], color='green',
               s=8, alpha=0.6, marker='s', label='Binary Sample')
    ax.set_xlabel(r'$s_1$', fontsize=18)
    ax.set_ylabel(r'$s_2$', fontsize=18)
    ax.set_title(type, fontsize=18)
    ax.legend()
    plt.savefig(path + 'generator/scRNA/' + parameters['dataset'] + '_cell_' +
                str(count) + '.png')
    plt.close()

# Plot per gene
csv_genes = pd.read_csv(file_marker_genes)
overlap_genes = csv_genes['overlapping_genes']
all_genes = dataset['gene_id_new'][:parameters['n_features']]
count = 0
th = data_origin.shape[0] * 0.10
for gene in overlap_genes[:100]:
    count +=1
    indx = np.where(all_genes == gene)[0]
    if len(indx) > 0:
        origin = data_origin[:, indx]
        generated = data_generate[:, indx]
        generated = generated[generated > 0]
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        sns.distplot(origin[origin > 0], kde=False, hist=True, ax=ax, bins=100,
                     label='Original', hist_kws={"alpha": 0.3, "color": "b"})
        if len(generated) > 0:
            sns.distplot(generated, kde=False, hist=True, ax=ax, bins=100,
                         label='Generated',
                         hist_kws={"alpha": 0.3, "color": "r"})
            ax.set_title('expression distribution for ' + gene, fontsize=18)
        else:
            ax.set_title('expression distribution for ' + gene +
                         ' (insufficient samples)', fontsize=18)

        ax.legend()
        plt.savefig(
            path + 'generator/scRNA/' + parameters['dataset'] + '_gene_' +
            str(count) + '.png')
        plt.close()
        if count == 34:
            stop = 1