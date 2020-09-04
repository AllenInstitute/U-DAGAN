from module.train import *

# Use GPU if available
device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Dictionary of the training parameters for FACS datatset
parameters = {'batch_size': 1000,  # batch size
            'num_epochs': 1000,  # number of epochs
            'learning_rate': 0.5e-4, # learning rate
            'num_z': 10, # latent space dimension
            'lambda': [2, 0., 0., 0.5], # weights of the augmenter loss
            'dataset': 'FACS', # dataset, i.e. {'MNIST', 'FACS'}
            'dataset_file': 'Mouse-V1-ALM-20180520_cpmtop10k_cpm_withCL.mat',
            'n_features': 5000,
            'n_smp': 20, # number of generated samples
            'remove_nonneuron': 'True', # remove non-neuronal cells
            'remove_CR_Meis2': 'False', # remove "CR Lhx5" & "Meis2 Adamts19"
            'initial_w': 'False', # initial weights
            'save': 'True', # saving flag
            }


train_vaegan(parameters, device)
