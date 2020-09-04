from module.train import *

# Use GPU if available
device = torch.device("cuda:1" if(torch.cuda.is_available()) else "cpu")
print(device, " will be used.\n")

# Dictionary of the training parameters for MNIST datatset
# parameters = {'batch_size': 128,  # batch size
#             'num_epochs': 100,  # number of epochs
#             'learning_rate': 1e-4, # learning rate
#             'alpha': 0.25,  # triplet loss hyperparameter
#             'num_z': 10, # latent space dimension
#             'num_n': 12, # noise dimension
#             'lambda': [0.5, 0., 0.5], # weights of the augmenter loss
#             'dataset': 'MNIST', # dataset, i.e. {'MNIST', 'FACS'}
#             'dataset_file': '',
#             'n_features': [],
#             'n_arm': 20, # number of augmented samples
#             'initial_w': 'False', # initial weights
#             'save': 'True', # saving flag
#             }


# Dictionary of the training parameters for FACS datatset
parameters = {'batch_size': 1000,  # batch size
            'num_epochs': 1000,  # number of epochs
            'learning_rate': 0.5e-4, # learning rate
            'alpha': 0.05,  # triplet loss hyperparameter
            'num_z': 10, # latent space dimension
            'num_n': 50, # noise dimension
            'lambda': [1, 0.5, 0.1, 0.5], # weights of the augmenter loss
            'dataset': 'FACS', # dataset, i.e. {'MNIST', 'FACS'}
            'dataset_file': 'Mouse-V1-ALM-20180520_cpmtop10k_cpm_withCL.mat',
            'n_features': 5000,
            'n_arm': 20, # number of augmented samples
            'initial_w': 'False', # initial weights
            'save': 'True', # saving flag
            }


train_udagan(parameters, device)
