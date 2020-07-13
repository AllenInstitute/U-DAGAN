import torch, pickle
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
import time, math
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import gridspec, cm

# Define current_time
current_time = time.strftime('%Y-%m-%d-%H-%M-%S')
eps = 1e-8
EPS = 1e-12
device_num = 1


class augmentor(nn.Module):

    def __init__(self, input_dim, latent_dim, p_drop, n_aug):
        super(augmentor, self).__init__()
        self.dp = nn.Dropout(p_drop)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.n_aug = n_aug

        self.fce1 = nn.Linear(input_dim, 512)
        self.fce2 = nn.Linear(self.fce1.out_features, self.fce1.out_features
                              // 2)
        self.fce3 = nn.Linear(self.fce2.out_features, self.fce2.out_features
                              // 2)
        self.fce4 = nn.Linear(self.fce3.out_features, latent_dim)
        self.fce5 = nn.Linear(self.fce3.out_features, latent_dim)

        self.fcdz1 = nn.ModuleList([nn.Linear(latent_dim, latent_dim) for a
                                    in range(self.n_aug)])
        self.fcdz2 = nn.ModuleList([nn.Linear(2*latent_dim, 2*latent_dim) for a
                                    in range(self.n_aug)])
        self.fcdz3 = nn.ModuleList([nn.Linear(2*latent_dim, latent_dim) for a
                                    in range(self.n_aug)])
        self.fcdz4 = nn.ModuleList([nn.Linear(2*latent_dim, latent_dim) for a
                                    in range(self.n_aug)])
        self.fcd1 = nn.Linear(latent_dim, 256)
        self.fcd2 = nn.Linear(self.fcd1.out_features, self.fcd1.out_features
                              * 2)
        self.fcd3 = nn.Linear(self.fcd2.out_features, self.fcd2.out_features
                              * 2)
        self.fcd4 = nn.Linear(self.fcd3.out_features, input_dim)

    def reparam_trick(self, mu, sigma):
        """
        Generate samples from a normal distribution for reparametrization trick.

        input args
            mu: mean of the Gaussian distribution for q(s|z,x) = N(mu, sigma^2*I).
            log_sigma: log of variance of the Gaussian distribution for
                       q(s|z,x) = N(mu, sigma^2*I).

        return
            a sample from Gaussian distribution N(mu, sigma^2*I).
        """
        eps = Variable(torch.cuda.FloatTensor(sigma.size()).normal_())
        return eps.mul(sigma).add(mu)


    def encoder(self, x):
        h1 = self.relu(self.fce1(self.dp(x)))
        h2 = self.relu(self.fce2(h1))
        h3 = self.relu(self.fce3(h2))
        return self.fce4(h3), self.sigmoid(self.fce5(h3))

    def forward(self, x, z, a):
        x_mu, _ = self.encoder(x)
        z = F.leaky_relu(self.fcdz1[a](z), 0.2)
        y = torch.cat((x_mu, z), dim=1)
        y = self.relu(self.fcdz2[a](y))
        mu_ = self.fcdz3[a](y)
        sigma_ = self.sigmoid(self.fcdz4[a](y))
        x_ = self.reparam_trick(mu_, sigma_)
        h1 = F.leaky_relu(self.fcd1(x_), 0.2)
        h2 = F.leaky_relu(self.fcd2(h1), 0.2)
        h3 = F.leaky_relu(self.fcd3(h2), 0.2)
        return mu_, torch.log(sigma_ + eps), torch.tanh(self.fcd4(h3))


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features // 2)
        self.fc5 = nn.Linear(self.fc4.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        z_d = F.leaky_relu(self.fc4(x), 0.2)
        x = F.dropout(z_d, 0.3)
        return z_d, torch.sigmoid(self.fc5(x))


def run_aan(batch_size=128,
            n_aug=2,
            p_drop=0.5,
            latent_dim=1,
            n_epoch=5000,
            weights=np.array([0.1, 1, 1, 1, 1]),
            lr=0.0001,
            folder='',
            training_flag=True,
            trained_models='',
            save_flag=True,
            device='cuda'):
    """
    Trains the adversarial augmentation neural network.

    input args
        batch_size: size of batches.
        p_drop: dropout probability at the first layer of the network.
        latent_dim: dimension of the continuous latent variable.
        n_epoch: number of training epochs.
        weights: array of objectives function weights including generator,
                 encoder, and discriminator in the loss function.
        folder: the directory used for saving the results.
        training_flag: a boolean variable that is True during training a VAE
                        model and is False during reloading a pre-trained model.
        trained_model: the path of the pre-trained network.
        save: a boolean variable for saving the test results.
        device: selected GPU(s).

    return
        data_file_id: file of the test results, it the save flag is Ture.
    """
    # torch.manual_seed(cvset)
    torch.cuda.set_device(device_num)
    print(torch.cuda.get_device_name(torch.cuda.current_device()))

    # set random rotation and noise in the image
    kwargs = {'num_workers': 2, 'pin_memory': True} if device else {}
    train_set = datasets.MNIST('./data/mnist/', train=True,
                               download=True,
                               transform=transforms.ToTensor())

    test_set = datasets.MNIST('./data/mnist/', train=False,
                              transform=transforms.ToTensor())

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             drop_last=True, **kwargs)

    pixel_size_x, pixel_size_y = train_set.data.size()[1:]
    input_dim = pixel_size_x * pixel_size_y
    a_model = augmentor(input_dim=input_dim,
                      latent_dim=latent_dim,
                      p_drop=p_drop,
                      n_aug=n_aug).cuda(device_num)
    # g_model = augmentor(input_dim=2 * latent_dim,
    #                     output_dim=input_dim).cuda(device_num)
    d_model = discriminator(input_dim=input_dim).cuda(device_num)

    train_loss_d = np.zeros(n_epoch)
    train_loss_a = np.zeros(n_epoch)
    train_loss_g = np.zeros(n_epoch)
    train_loss_e = np.zeros(n_epoch)
    train_loss_KL = np.zeros(n_epoch)

    # define optimizers
    # e_optimizer = torch.optim.Adam(e_model.parameters(), lr=lr)
    # g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    a_optimizer = torch.optim.Adam(a_model.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=lr)

    # define loss functions
    loss_bce = nn.BCELoss()
    loss_mse = nn.MSELoss()

    if training_flag:
        # Model training
        print("Training...")

        for epoch in range(n_epoch):
            t0 = time.time()
            batch_train_loss_d = 0
            batch_train_loss_g = 0
            batch_train_loss_e = 0
            batch_train_loss_a = 0
            batch_train_loss_KL = 0
            # training
            for batch_indx, (data, _), in enumerate(train_loader):

                # discriminator training
                d_optimizer.zero_grad()
                loss_d_samp = []
                z_dfake, y_fake = [None] * n_aug, [None] * n_aug
                # for data in train_loader:
                data = Variable(data).cuda(device_num)
                data = data.view(-1, input_dim)
                l_real = Variable(torch.ones(batch_size)).cuda(device_num)
                l_fake = Variable(torch.zeros(batch_size)).cuda(device_num)
                # x_low = e_model(data)
                for a in range(n_aug):
                    z = Variable(torch.randn(batch_size, latent_dim)).cuda(
                        device_num)
                    _, _, data_aug = a_model(data, z, a)
                    _, y_fake[a] = d_model(data_aug)
                _, y_real = d_model(data)

                for a in range(n_aug):
                    loss_d_samp.append(loss_bce(y_fake[a], l_fake))
                loss_d = loss_bce(y_real, l_real) + sum(loss_d_samp)
                loss_d.backward()
                d_optimizer.step()

                # augmenter training
                a_optimizer.zero_grad()
                loss_g_samp, loss_e_samp = [], []
                data_augmented = [None] * n_aug
                z_dfake, y_fake = [None] * n_aug, [None] * n_aug
                mu, log_sigma = [None] * n_aug, [None] * n_aug
                # for data in train_loader:
                data = Variable(data).cuda(device_num)
                data = data.view(-1, input_dim)
                l_real = Variable(torch.ones(batch_size)).cuda(device_num)
                # x_low = e_model(data)
                for a in range(n_aug):
                    z = Variable(torch.randn(batch_size, latent_dim)).cuda(
                        device_num)
                    mu[a], log_sigma[a], data_augmented[a] = a_model(data, z, a)
                    z_dfake[a], y_fake[a] = d_model(data_augmented[a])
                z_dreal, _ = d_model(data)

                for a in range(n_aug):
                    loss_g_samp.append(loss_bce(y_fake[a], l_real))
                    loss_e_samp.append(weights[0] * (data_augmented[a] - data).pow(
                        2).mean() + weights[1] * (z_dfake[a] - z_dreal).pow(
                        2).mean())

                loss_g = sum(loss_g_samp)
                loss_e = sum(loss_e_samp)
                loss_KL = 0.5 * torch.mean(-1 + log_sigma[1] - log_sigma[0] +
                                           (log_sigma[0].exp() +
                                           (mu[0] - mu[1]).pow(2)) /
                                            log_sigma[1].exp(), dim=0).mean()
                loss_dist = (mu[0] - mu[1]).pow(2).mean()
                loss_a = weights[2] * loss_g + weights[3] * loss_e \
                         - weights[4] * loss_dist
                loss_a.backward()
                a_optimizer.step()

                batch_train_loss_d += loss_d.item()
                batch_train_loss_g += loss_g.item()
                batch_train_loss_e += loss_e.item()
                batch_train_loss_a += loss_a.item()
                batch_train_loss_KL += loss_KL.item()


            train_loss_d[epoch] = batch_train_loss_d / (batch_indx+1) #len(train_loader.dataset)
            train_loss_g[epoch] = batch_train_loss_g / (batch_indx+1) #len(train_loader.dataset)
            train_loss_e[epoch] = batch_train_loss_e / (batch_indx+1) #len(train_loader.dataset)
            train_loss_a[epoch] = batch_train_loss_a / (batch_indx+1) #len(train_loader.dataset)
            train_loss_KL[epoch] = batch_train_loss_KL / (batch_indx + 1)  #len(train_loader.dataset)

            print('====> Epoch:{}, DLoss: {:.4f}, GLoss: {'
                  ':.4f}, ELoss: {:.4f}, ALoss: {:.4f}, KL: {:.4f}, Elapsed '
                  'Time:{'
                  ':.2f}'.format(epoch, train_loss_d[epoch], train_loss_g[epoch],
                train_loss_e[epoch], train_loss_a[epoch], train_loss_KL[epoch],
                                 time.time() - t0))

        # torch.save(e_model.state_dict(), folder +
        #            '/model/encoder_model_' + current_time)
        # torch.save(g_model.state_dict(), folder +
        #            '/model/generator_model_' + current_time)
        torch.save(a_model.state_dict(), folder +
                   '/model/augmenter_model_' + current_time)
        torch.save(d_model.state_dict(), folder +
                   '/model/discriminator_model_' + current_time)
    else:
        # if you wish to load another model for evaluation
        d_model.load_state_dict(torch.load(trained_models[0]))
        a_model.load_state_dict(torch.load(trained_models[1]))
        # e_model.load_state_dict(torch.load(trained_models[1]))
        # g_model.load_state_dict(torch.load(trained_models[2]))

    # Evaluate the trained model
    d_model.eval()
    a_model.eval()
    # e_model.eval()
    # g_model.eval()

    n = 10

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data_augmented = [None] * n_aug
            z_dfake, y_fake = [None] * n_aug, [None] * n_aug
            data = Variable(data).cuda(device_num)

            for a in range(n_aug):
                z = Variable(torch.randn(batch_size, latent_dim)).cuda(
                    device_num)
                _, _, data_augmented[a] = a_model(data.view(-1, input_dim),
                                                  z, a)

            comparison = data[:n]
            for a in range(n_aug):
                comparison = torch.cat([comparison,
                                       data_augmented[a].view(data.size(0), 1,
                                                              pixel_size_x,
                                                              pixel_size_y)[
                                       :n]])
            if save_flag:
                save_image(comparison.data.cpu(),
                           folder +
                           '/augmented_samples/sample_'
                           + str(i) + '_nAug_' + str(n_aug) + '.png',
                           nrow=n)
    if training_flag:
        plt.figure()
        plt.plot(range(n_epoch), train_loss_d, '--', label='loss_d')
        plt.plot(range(n_epoch), train_loss_a, '-.', label='loss_a')
        plt.legend()
        if save_flag:
            plt.savefig(folder + '/model/training_loss.png')

    # save data
    data_file_id = folder + '/model/data' + current_time
    if save_flag:
        save_file(data_file_id,
                  train_loss_d=train_loss_d,
                  train_loss_a=train_loss_a,
                  train_loss_g=train_loss_g,
                  train_loss_e=train_loss_e)

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


def func(pct, allvals):
    absolute = int(np.sum(allvals) / pct)
    return "{:.1f}%".format(pct)

