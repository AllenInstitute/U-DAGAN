import scipy.io as sio
import numpy as np
import os, pickle, glob
from module.aan import run_aan


# set the parameter of the AE model
# set the parameter of the AE model

p_drop = 0.2
latent_dim = 100
n_epoch = 500
batch_size = 256
training_flag = False
save_flag = True
n_aug = 10
lr = 0.0001
weights = [1., 1., 1., 1., 1.]

if training_flag:
    main_path = './results/mnist/'
else:
    main_path = '/results/mnist/'

for run in range(1):
    file_name = str(run) + \
                '_det_zdim_' + str(latent_dim) + \
                '_drop_' + str(p_drop * 100) + \
                '_WpixLoss_' + str(weights[0]) + \
                '_WlayerLoss_' + str(weights[1]) + \
                '_WgenLoss_' + str(weights[2]) + \
                '_WencLoss_' + str(weights[3]) + \
                '_WdistLoss_' + str(weights[4]) + \
                '_lr_' + str(lr) + \
                '_bSize' + str(batch_size) + \
                '_e_' + str(n_epoch)

    if training_flag:
        os.makedirs(main_path + file_name, exist_ok=True)
        os.makedirs(main_path + file_name + '/augmented_samples', exist_ok=True)
        os.makedirs(main_path + file_name + '/model', exist_ok=True)

        result_folder = main_path + file_name
        trained_models = result_folder + '/model/' + ''

    else:
        result_folder = main_path + file_name
        current_dir = os.getcwd()
        aug_model = glob.glob(current_dir + result_folder +
                             '/model/augmenter_model_*')[0]
        disc_model = glob.glob(current_dir + result_folder +
                              '/model/discriminator_model_*')[0]
        trained_models, trained_model = [None] * 2, [None] * 2
        trained_model[0] = disc_model[disc_model.find('model'):]
        trained_model[1] = aug_model[aug_model.find('model'):]
        result_folder = '.' + result_folder
        trained_models[0] = result_folder + '/' + trained_model[0]
        trained_models[1] = result_folder + '/' + trained_model[1]


    # run mixVAE - MNIST dataset
    data_file_id = run_aan(batch_size=batch_size,
                            n_epoch=n_epoch,
                            latent_dim=latent_dim,
                            n_aug=n_aug,
                            p_drop=p_drop,
                            weights= weights,
                            save_flag=save_flag,
                            training_flag=training_flag,
                            lr=lr,
                            trained_models=trained_models,
                            folder=result_folder)
