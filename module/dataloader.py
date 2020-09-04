import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.utils.data import DataLoader, TensorDataset
import scipy.io as sio
import numpy as np

# Directory containing the data.
root = './data/'

def get_data(dataset, batch_size, file, n_feature, training=True,
             remove_nonneuron=True, remove_CR_Meis2=False):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        tensor_data = dsets.MNIST(root + 'mnist/', train='train',
                                download=True, transform=transforms.ToTensor())
        data = []

    if dataset == 'FACSy':
        # load the cells data .mat file
        data = sio.loadmat(root + 'facs/' + file, squeeze_me=True)
        # remove measurement labeled "Low Quality"
        ind = np.where((data['class_label'] == 'Low Quality'))
        ref_len = len(data['class_label'])
        all_key = list(data.keys())
        for key in all_key:
            if len(data[key]) >= ref_len:
                data[key] = np.delete(data[key], ind, axis=0)

        # remove measurement labeled "Non-Neuronal"
        if remove_nonneuron:
            ind = np.where((data['class_label'] == 'Non-Neuronal'))
            ref_len = len(data['class_label'])
            all_key = list(data.keys())
            for key in all_key:
                if len(data[key]) >= ref_len:
                    data[key] = np.delete(data[key], ind, axis=0)

        # remove cells "CR Lhx5" and "Meis2 Adamts19"
        if remove_CR_Meis2:
            ind = np.where((data['cluster'] == 'CR Lhx5'))
            ref_len = len(data['cluster'])
            all_key = list(data.keys())
            for key in all_key:
                if len(data[key]) >= ref_len:
                    data[key] = np.delete(data[key], ind, axis=0)
            ind = np.where((data['cluster'] == 'Meis2 Adamts19'))
            ref_len = len(data['cluster'])
            all_key = list(data.keys())
            for key in all_key:
                if len(data[key]) >= ref_len:
                    data[key] = np.delete(data[key], ind, axis=0)

        data_cpm = data['log1p'][:, :n_feature]
        data_cpm_bin = np.copy(data_cpm)
        data_cpm_bin[data_cpm_bin > 0] = 1
        # data_cpm_norm = data_cpm / np.expand_dims(max_exp, axis=1).repeat(
        #     n_feature, axis=1)
        data_cpm_troch = torch.FloatTensor(data_cpm)
        data_cpm_bin_troch = torch.FloatTensor(data_cpm_bin)
        all_ind_torch = torch.FloatTensor(range(len(data_cpm)))
        tensor_data = TensorDataset(data_cpm_troch, data_cpm_bin_troch,
                                    all_ind_torch)

    # Create dataloader.
    if training:
        dataloader = DataLoader(tensor_data,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=True)
    else:
        dataloader = DataLoader(tensor_data,
                                batch_size=batch_size,
                                shuffle=True,
                                drop_last=False)

    return dataloader, data