import torch, pickle
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import time, math
from sklearn.utils import shuffle
from torch.utils.data import DataLoader, TensorDataset
from operator import itemgetter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import gridspec, cm

# Define current_time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
eps = 1e-8
device_num = 1


class singleAE(nn.Module):
    """
    Class for the neural network module for mixture of continuous and
    discrete random variables. The module contains an VAE using
    Gumbel-softmax distribution for the categorical and reparameterization
    for continuous latent variables.
    The default setting of this network is for FACS dataset. If you want to
    use another dataset, you need to modify the network's parameters.

    Methods
        encode: encoder network.
        intermed: the intermediate layer for combining categorical and
                  continuous RV.
        forward: decoder network.
    """
    def __init__(self, input_dim, fc_dim, p_drop, latent_dim):
        """
        Class instantiation.

        input args
            input_dim: input dimension (size of the input layer).
            fc_dim: dimension of the lower representation of the input data.
            p_drop: dropout probability at the first layer.
            state_latent_dim: dimension of the continuous latent variable.
            n_categories: number of categories of the latent variables.
            loss_type: type of reconstruction loss function that can, "mse",
            "zinb", or "gamma".
        """
        super(singleAE, self).__init__()
        self.input_dim = input_dim
        self.fc_dim = fc_dim
        self.latent_dim = latent_dim
        self.dp = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)
        self.fc3 = nn.Linear(fc_dim, fc_dim)
        self.fc4 = nn.Linear(fc_dim, fc_dim)
        self.fc5 = nn.Linear(fc_dim, latent_dim)
        self.fc6 = nn.Linear(latent_dim, fc_dim)
        self.fc7 = nn.Linear(fc_dim, fc_dim)
        self.fc8 = nn.Linear(fc_dim, fc_dim)
        self.fc9 = nn.Linear(fc_dim, fc_dim)
        self.fc10 = nn.Linear(fc_dim, input_dim)

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

        self.batch = nn.BatchNorm1d(num_features=latent_dim, eps=1e-10,
                                    momentum=0.01, affine=False)


    def encode(self, x):
        h1 = self.relu(self.fc1(self.dp(x)))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        h4 = self.relu(self.fc4(h3))
        return self.batch(self.fc5(h4))

    def decode(self, s):
        h6 = self.relu(self.fc6(s))
        h7 = self.relu(self.fc7(h6))
        h8 = self.relu(self.fc8(h7))
        h9 = self.relu(self.fc9(h8))
        return self.relu(self.fc10(h9))

    def forward(self, x):
        x_low = self.encode(x)
        return self.decode(x_low), x_low

class myVAE(nn.Module):
    def __init__(self, latent_dim, input_dim=5000, n_dim=500, p_drop=0.5):
        super(myVAE, self).__init__()

        self.dp = nn.Dropout(p_drop)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc5 = nn.Linear(n_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fcs = nn.Linear(self.fc5.out_features, latent_dim)
        self.batch_fcs = nn.BatchNorm1d(num_features=self.fcs.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc6 = nn.Linear(self.fcs.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc10 = nn.Linear(self.fc9.out_features, self.fc9.out_features)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features,
                                        eps=1e-10, momentum=0.01, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x):

        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        x = F.relu(self.batch_fc3(self.fc3(x)))
        x = F.relu(self.batch_fc4(self.fc4(x)))
        x = F.relu(self.batch_fc5(self.fc5(x)))
        s = self.batch_fcs(self.fcs(x))
        x = F.relu((self.fc6(s)))
        x = F.relu((self.fc7(x)))
        x = F.relu((self.fc8(x)))
        x = F.relu((self.fc9(x)))
        x = F.relu((self.fc10(x)))

        return F.relu(self.fc11(x)), s

def loss_function_singleAE(recon_x, x):
    """
    loss function of mix-VAE network only including continuous RV:
    loss = -E_{q(s,z|x)}[log(p(x|z,s))] +
           E_{q(z|z)}[KL(q(s|z,x) || p(s|z))]
    The default setting of this network is for FACS dataset, using MSE for the
    first term of the objective function, log(p(x|z,s)).

   input args
        recon_x: a list including the reconstructed images.
        x: a list includes original input data.

    return
        loss values of mix-VAE network i.e., l_vae = l_rec + KLD_cont
    """

    return F.mse_loss(recon_x, x, reduction='mean')


def data_gen(dataset, cvset=0, n_genes=5000, train_size=22000, test_size=2000):


    train_cpm, val_cpm, train_ind, val_ind = train_test_split(
        dataset['log1p'][:, :n_genes], np.arange(dataset['log1p'].shape[0]),
        train_size=train_size,
        test_size=dataset['log1p'].shape[0] - train_size)

    train_cpm = train_cpm[:-test_size, :]
    train_ind = train_ind[:-test_size]
    all_cpm = dataset['log1p'][:, :n_genes]

    return train_cpm, val_cpm, train_ind, val_ind, all_cpm


def dataIO_cpm(dataset, cvset, n_genes, train_per, val_per, test_per):
    """
    Shuffles and splits the original dataset for train and validation.

    input args
        dataset: the entire dataset used for training and validation.
        cvset: seed of random generator.
        n_genes: number of genes (features) used for training.
        train_size: size of the training data component.
        val_size: size of the validation data component.
        test_size: size of the test data component.

    return
        train_cpm: training data component.
        val_cpm: validation data component.
        test_cpm: test data component.
        train_ind: index of training data component in the original dataset.
        val_ind: index of validation data component in the original dataset.
        test_ind: index of test data component in the original dataset.
    """
    label_list = np.unique(dataset['class_label'])
    data_size = len(dataset['class_label'])
    dataset['classID'] = -np.ones(data_size)
    data_per_class = np.zeros(len(label_list))
    id = 0
    for label in label_list:
        dataset['classID'][np.where(dataset['class_label'] == label)] = id
        data_per_class[id] = np.int(np.sum(dataset['class_label'] == label))
        id += 1

    data_cpm = dataset['log1p'][:, :n_genes]
    label_cpm = dataset['classID']
    train_cpm = np.empty((0, n_genes))
    val_cpm = np.empty((0, n_genes))
    test_cpm = np.empty((0, n_genes))
    train_label = np.empty(0)
    val_label = np.empty(0)
    test_label = np.empty(0)
    train_ind = np.empty(0)
    val_ind = np.empty(0)
    test_ind = np.empty(0)
    id = 0
    for label in label_list:
        indx = np.where(label_cpm == id)
        class_data = np.squeeze(data_cpm[indx, :], axis=0)
        train_size = np.int(train_per * data_per_class[id])
        test_size = np.int(test_per * data_per_class[id])
        val_size = np.int(data_per_class[id] - train_size - test_size)
        train_cpm = np.append(train_cpm, class_data[:train_size, :], axis=0)
        # train_cpm.append(class_data[:train_size,:])
        val_cpm = np.append(val_cpm, class_data[
                                     train_size:train_size + val_size, :],
                            axis=0)
        # val_cpm.append(class_data[train_size:train_size+val_size,:])
        test_cpm = np.append(test_cpm, class_data[
                                       train_size + val_size:, :], axis=0)
        # test_cpm.append(class_data[train_size+val_size:,:])
        train_label = np.append(train_label, id * np.ones(train_size), axis=0)
        val_label = np.append(val_label, id * np.ones(val_size), axis=0)
        test_label = np.append(test_label, id * np.ones(test_size), axis=0)
        train_ind = np.append(train_ind, indx[0][:train_size], axis=0)
        val_ind = np.append(val_ind, indx[0][train_size:train_size + val_size],
                            axis=0)
        test_ind = np.append(test_ind, indx[0][train_size + val_size:], axis=0)
        id += 1
    # shuffle train, validation, and test sets
    train_cpm, train_shuff_ind = shuffle(train_cpm, range(train_cpm.shape[0]))
    val_cpm, val_shuff_ind = shuffle(val_cpm, range(val_cpm.shape[0]))
    test_cpm, test_shuff_ind = shuffle(test_cpm, range(test_cpm.shape[0]))
    train_label = train_label[train_shuff_ind].astype(int)
    val_label = val_label[val_shuff_ind].astype(int)
    test_label = test_label[test_shuff_ind].astype(int)
    train_ind = train_ind[train_shuff_ind].astype(int)
    val_ind = val_ind[val_shuff_ind].astype(int)
    test_ind = test_ind[test_shuff_ind].astype(int)
    return data_cpm, train_cpm, val_cpm, test_cpm, train_ind, val_ind, \
           test_ind, \
           train_label, val_label, test_label



def run_singleAE(dataset,
                cvset=0,
                batch_size=128,
                n_features=5000,
                train_size=0.85,
                val_size=0.05,
                test_size=.1,
                p_drop=0.5,
                latent_dim=5,
                fc_dim=10,
                n_epoch=5000,
                lr=0.001,
                folder='',
                training_flag=True,
                trained_model='',
                save_flag=True):
    """
    Trains the mix-VAE neural network.

    input args
        dataset: the entire dataset used for training and validation.
        cvset: seed of random generator.
        batch_size: size of batches.
        n_features: number of features (columns) of dataset used in training.
        train_size: size of the training data component.
        val_size: size of the validation data component.
        test_size: size of the test data component.
        p_drop: dropout probability at the first layer of the network.
        latent_dim: dimension of the continuous latent variable.
        n_categorical: number of categories of the latent variables.
        fc_dim: dimension of the lower representation of the input data.
        n_epoch: number of training epochs.
        anneal_rate: annealing rate of temperature parameter.
        temp_min: minimum value of temperature parameter.
        temp_0: initial temperature value.
        on_hot_flag: a boolean flag, which defines ST Gumbel or regular method.
        model_id: the type of the VAE used in the output file's id.
        folder: the directory used for saving the results.
        training_flag: a boolean variable that is True during training a VAE
                        model and is False during reloading a pre-trained model.
        trained_model: the path of the pre-trained network.
        save: a boolean variable for saving the test results.
        device: selected GPU(s).

    return
        data_file_id: file of the test results, it the save flag is Ture.
    """
    # manually set the random seed
    # torch.manual_seed(cvset)
    torch.cuda.set_device(device_num)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    train_cpm, val_cpm, train_ind, val_ind, data_cpm = \
        data_gen(dataset=dataset, n_genes=n_features)

    # data_cpm, train_cpm, val_cpm, test_cpm, train_ind, val_ind, test_ind, \
    # train_label, val_label, test_label = dataIO_cpm(dataset,
    #                                                 cvset=cvset,
    #                                                 n_genes=n_features,
    #                                                 train_per=train_size,
    #                                                 val_per=val_size,
    #                                                 test_per=test_size)

    # max_exp = data_cpm.max(1)
    # data_cpm = data_cpm / np.expand_dims(max_exp, axis=1).repeat(
    #     n_features, axis=1)

    train_cpm_torch = torch.FloatTensor(train_cpm)
    train_ind_torch = torch.FloatTensor(train_ind)
    train_data = TensorDataset(train_cpm_torch,
                               train_ind_torch)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True)
    val_cpm_torch = torch.FloatTensor(val_cpm)
    val_ind_torch = torch.FloatTensor(val_ind)
    validation_data = TensorDataset(val_cpm_torch,
                                    val_ind_torch)
    validation_loader = DataLoader(validation_data,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True)
    test_cpm_torch = torch.FloatTensor(val_cpm)
    test_ind_torch = torch.FloatTensor(val_ind)
    test_data = TensorDataset(test_cpm_torch,
                              test_ind_torch)
    test_loader = DataLoader(test_data,
                             batch_size=1,
                             shuffle=True,
                             drop_last=True)
    data_cpm_troch = torch.FloatTensor(data_cpm)
    all_ind_torch = torch.FloatTensor(range(len(data_cpm)))
    all_data = TensorDataset(data_cpm_troch,
                             all_ind_torch)
    alldata_loader = DataLoader(all_data,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False)

    input_dim = train_cpm.shape[1]
    # model = singleAE(input_dim=input_dim,
    #                 fc_dim=fc_dim,
    #                 latent_dim=latent_dim,
    #                 p_drop=p_drop).cuda(device_num)

    model = myVAE(input_dim=input_dim,
                     n_dim=fc_dim,
                     p_drop=p_drop,
                     latent_dim=latent_dim).cuda(device_num)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss = np.zeros(n_epoch)
    validation_loss = np.zeros(n_epoch)

    learning_rate = np.zeros(n_epoch)
    # lr = 0.0001
    if training_flag:
        # Model training
        print("Training...")
        model.train()
        for epoch in range(n_epoch):
            train_loss_val = 0
            t0 = time.time()
            color_code = ["" for x in range(len(train_loader.dataset))]
            lowD_rep = np.zeros((len(train_loader.dataset), latent_dim))
            # training
            for batch_indx, (data, labels), in enumerate(train_loader):
                optimizer.zero_grad()
                # for data in train_loader:
                data = Variable(data).cuda(device_num)
                l = [int(lab) for lab in labels.numpy()]
                color_code[batch_indx * len(l):(batch_indx + 1) * len(l)] = \
                    dataset['cluster_color'][l]
                recon_batch, x_low = model(data)
                lowD_rep[batch_indx * x_low.size(0):(batch_indx + 1) *
                                x_low.size(0), :] = x_low.cpu().detach().numpy()
                loss =loss_function_singleAE(recon_batch, data)
                train_loss_val += loss.data.item()
                loss.backward()
                optimizer.step()

            train_loss[epoch] = train_loss_val / (batch_indx + 1)
            # validation
            val_cpm_torch = val_cpm_torch.cuda(device_num)
            recon_batch, _ = model(val_cpm_torch)
            loss = loss_function_singleAE(recon_batch, val_cpm_torch)
            validation_loss[epoch] = loss.data.item()

            # if epoch < 1000:
            #     if epoch % 10 == 0:
            #         plt.figure()
            #         plt.scatter(lowD_rep[:batch_indx*batch_size, 0],
            #                     lowD_rep[:batch_indx*batch_size, 1],
            #                     color=color_code[:batch_indx*batch_size],
            #                     s=2, alpha=0.8)
            #         plt.rc('axes',labelsize=14)
            #         plt.rc('xtick', labelsize=10)
            #         plt.rc('ytick', labelsize=10)
            #         plt.ylabel('dim2')
            #         plt.xlabel('dim1')
            #         plt.axis('square')
            #         if epoch > 30:
            #             plt.xlim([-3.5, 3.5])
            #             plt.ylim([-3.5, 3.5])
            #
            #         if save_flag:
            #             plt.savefig(
            #                 folder + '/lowD_frames/lowD_representation_dim_' + str(
            #                     latent_dim) +
            #                 '_epoch_' + str(epoch) + '.png', dpi=300)
            #         plt.close('all')
            # else:
            #     if epoch % 1000 == 0:
            #         plt.figure()
            #         plt.scatter(lowD_rep[:batch_indx * batch_size, 0],
            #                     lowD_rep[:batch_indx * batch_size, 1],
            #                     color=color_code[:batch_indx * batch_size],
            #                     s=2, alpha=0.8)
            #         plt.rc('axes', labelsize=14)
            #         plt.rc('xtick', labelsize=10)
            #         plt.rc('ytick', labelsize=10)
            #         plt.ylabel('dim2')
            #         plt.xlabel('dim1')
            #         plt.axis('square')
            #         plt.xlim([-3.5, 3.5])
            #         plt.ylim([-3.5, 3.5])
            #
            #         if save_flag:
            #             plt.savefig(
            #                 folder + '/lowD_frames/lowD_representation_dim_' + str(
            #                     latent_dim) +
            #                 '_epoch_' + str(epoch) + '.png', dpi=300)
            #         plt.close('all')

            print('====> Epoch:{}, Training Loss: {:.4f}, Validation Loss: {'
                  ':.4f}, Elapsed Time:{:.4f}'.format(epoch, train_loss[epoch],
                                                      validation_loss[epoch],
                                                      time.time() - t0))

        torch.save(model.state_dict(), folder +
                   '/model/singleAE_model_' + current_time)
    else:
        # if you whish to load another model for evaluation
        model.load_state_dict(torch.load(trained_model))
    # Evaluate the trained model
    model.eval()
    test_cpm_torch = test_cpm_torch.cuda(device_num)
    recon_batch, _ = model(test_cpm_torch)
    loss = loss_function_singleAE(recon_batch, test_cpm_torch)

    test_loss = loss.data.item() #/ len(test_cpm_torch)
    print('====> Test set loss: {:.4f}'.format(test_loss))

    data_low = np.zeros((len(alldata_loader.dataset), latent_dim))
    data_indx = np.zeros(len(alldata_loader.dataset))
    samp_clr = ["" for x in range(len(alldata_loader.dataset))]

    with torch.no_grad():
        for i, (data, labels) in enumerate(alldata_loader):
            data = Variable(data).cuda()
            _, tmp = model(data)
            l = [int(lab) for lab in labels.numpy()]
            samp_clr[i * batch_size:min((i + 1) * batch_size,
                                        len(alldata_loader.dataset))] = \
                dataset['cluster_color'][l]
            data_indx[i * batch_size:min((i + 1) * batch_size,
                                         len(alldata_loader.dataset))] = \
                labels.cpu().detach().numpy()
            data_low[i * batch_size:min((i + 1) * batch_size,
                                        len(alldata_loader.dataset)), :] = \
                tmp.cpu().detach().numpy()

    # plot 2D feature space
    plt.subplots()
    plt.scatter(data_low[:, 0], data_low[:, 1], color=samp_clr, s=1.5,
                alpha=0.8)
    plt.xlabel(r'$s_1$', fontsize=18)
    plt.ylabel(r'$s_2$', fontsize=18)
    plt.title('Latent Space', fontsize=16)
    if save_flag:
        plt.savefig(folder + '/lowD_feature_space_dim_' + str(latent_dim) +
                         '.png')

    # plot the learning curve of the network
    fig, ax = plt.subplots()
    ax.plot(range(n_epoch), train_loss)
    ax.set_xlabel('# epoch', fontsize=16)
    ax.set_ylabel('loss value', fontsize=16)
    ax.set_title('Reconstruction error, Single AE for d=' + str(latent_dim))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if save_flag:
        ax.figure.savefig(folder + '/model/recons_error_ld_' + str(latent_dim)
                          + '_e_' + str(n_epoch) + '.png')
    # save data
    data_file_id = folder + '/model/data_' + current_time
    if save_flag:
        save_file(data_file_id,
                  train_loss=train_loss,
                  validation_loss=validation_loss,
                  test_loss=test_loss,
                  fc_min=fc_dim,
                  latent_dim=latent_dim)

    if save_flag:
        save_file(folder + '/model/FACS_5D_' + current_time,
                  gene_val=data_low,
                  indx=data_indx)

    return data_file_id

def save_file(fname, **kwargs):
    """
    Save data as a .p file using pickle.

    input args
        fname: the path of the pre-trained network.
        kwarg: keyword arguments for input variables e.g., x=[], y=[], etc.
    """
    f = open(fname + '.p', "wb")
    data = {}
    for k, v in kwargs.items():
        data[k] = v
    pickle.dump(data, f)
    f.close()

def load_file(fname):
    """
    load data .p file using pickle. Make sure to use the same version of
    pcikle used for saving the file

    input args
        fname: the path of the pre-trained network.

    return
        data: a dictionary including the save dataset
    """
    data = pickle.load(open(fname + '.p', "rb"))
    return data


