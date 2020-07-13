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
device_num = 0


class generator(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, input_dim)

    # forward method
    def forward(self, x, z):
        x = torch.cat((x, z), dim=1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class discriminator(nn.Module):
    def __init__(self, input_dim):
        super(discriminator, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features//2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features//2)
        self.fc4 = nn.Linear(self.fc3.out_features, 1)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))


def noise_sample(n_dis_c, dis_c_dim, n_con_c, n_z, batch_size, device):
    """
    Sample random noise vector for training.
    INPUT
    --------
    n_dis_c : Number of discrete latent code.
    dis_c_dim : Dimension of discrete latent code.
    n_con_c : Number of continuous latent code.
    n_z : Dimension of iicompressible noise.
    batch_size : Batch Size
    device : GPU/CPU
    """

    z = torch.randn(batch_size, n_z, 1, 1, device=device)

    idx = np.zeros((n_dis_c, batch_size))
    if (n_dis_c != 0):
        dis_c = torch.zeros(batch_size, n_dis_c, dis_c_dim, device=device)

        for i in range(n_dis_c):
            idx[i] = np.random.randint(dis_c_dim, size=batch_size)
            dis_c[torch.arange(0, batch_size), i, idx[i]] = 1.0

        dis_c = dis_c.view(batch_size, -1, 1, 1)

    if (n_con_c != 0):
        # Random uniform between -1 and 1.
        con_c = torch.rand(batch_size, n_con_c, 1, 1, device=device) * 2 - 1

    noise = z
    if (n_dis_c != 0):
        noise = torch.cat((z, dis_c), dim=1)
    if (n_con_c != 0):
        noise = torch.cat((noise, con_c), dim=1)

    return noise, idx

def run_infogan(batch_size=128,
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
            device='cuda',
            cuda=True):
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

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])])

    train_set = datasets.MNIST('./data/mnist/', train=True,
                               download=True,
                               transform=transform)

    test_set = datasets.MNIST('./data/mnist/', train=False,
                              transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, drop_last=True, **kwargs)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                             drop_last=True, **kwargs)

    pixel_size_x, pixel_size_y = train_set.data.size()[1:]
    input_dim = pixel_size_x * pixel_size_y
    g_model = generator(input_dim=input_dim,
                      latent_dim=latent_dim).cuda(device_num)
    d_model = discriminator(input_dim=input_dim).cuda(device_num)

    train_loss_d = np.zeros(n_epoch)
    train_loss_a = np.zeros(n_epoch)
    train_loss_g = np.zeros(n_epoch)
    train_loss_e = np.zeros(n_epoch)

    # define optimizers
    # e_optimizer = torch.optim.Adam(e_model.parameters(), lr=lr)
    # g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    g_optimizer = torch.optim.Adam(g_model.parameters(), lr=lr)
    d_optimizer = torch.optim.Adam(d_model.parameters(), lr=lr)

    # define loss functions
    loss_bce = torch.nn.BCELoss()
    loss_mse = torch.nn.MSELoss()

    if training_flag:
        # Model training
        print("Training...")

        for epoch in range(n_epoch):
            t0 = time.time()
            batch_train_loss_d = 0
            batch_train_loss_g = 0
            batch_train_loss_e = 0
            batch_train_loss_a = 0
            # training
            for batch_indx, (data, _), in enumerate(train_loader):

                # for data in train_loader:
                data = Variable(data).cuda(device_num)
                data_r = data.view(-1, input_dim)
                l_real = Variable(torch.ones(batch_size, 1)).cuda(device_num)
                l_fake = Variable(torch.zeros(batch_size, 1)).cuda(device_num)
                # x_low = e_model(data)
                # discriminator training
                d_optimizer.zero_grad()

                z = Variable(torch.randn(batch_size, latent_dim)).cuda(device_num)
                data_aug = g_model(z)
                y_fake = d_model(data_aug)
                y_real = d_model(data_r)

                loss_d = (loss_bce(y_real, l_real) + loss_bce(y_fake, l_fake))/2
                loss_d.backward()
                d_optimizer.step()

                # generator training
                g_optimizer.zero_grad()
                z = Variable(torch.randn(batch_size, latent_dim)).cuda(device_num)
                data_augmented = g_model(z)
                # print(data_augmented)
                y_fake = d_model(data_augmented)
                loss_g = loss_bce(y_fake, l_real)
                loss_g.backward()
                g_optimizer.step()

                batch_train_loss_d += loss_d.item()
                batch_train_loss_g += loss_g.item()

            train_loss_d[epoch] = batch_train_loss_d/ (batch_indx+1) #len(
            # train_loader.dataset)
            train_loss_g[epoch] = batch_train_loss_g/ (batch_indx+1) #len(
            # train_loader.dataset)


            print('====> Epoch:{}, DLoss: {:.4f}, GLoss: {'
                  ':.4f}, Elapsed Time:{'
                  ':.2f}'.format(epoch, train_loss_d[epoch], train_loss_g[epoch],
                time.time() - t0))

            if epoch > n_epoch - 2:
                n = 10
                comparison = torch.cat([data[:n],
                                        data_augmented.view(data.size(0), 1,
                                                            pixel_size_x,
                                                            pixel_size_y)[:n]])
                if save_flag:
                    save_image(comparison.data.cpu(),
                               folder +
                               '/augmented_samples/training_sample_'
                               + '_nAug_' + str(n_aug) + '.png',
                               nrow=n)


        # torch.save(e_model.state_dict(), folder +
        #            '/model/encoder_model_' + current_time)
        # torch.save(g_model.state_dict(), folder +
        #            '/model/generator_model_' + current_time)
        torch.save(g_model.state_dict(), folder +
                   '/model/augmenter_model_' + current_time)
        torch.save(d_model.state_dict(), folder +
                   '/model/discriminator_model_' + current_time)
    else:
        # if you wish to load another model for evaluation
        d_model.load_state_dict(torch.load(trained_models[0]))
        g_model.load_state_dict(torch.load(trained_models[1]))
        # e_model.load_state_dict(torch.load(trained_models[1]))
        # g_model.load_state_dict(torch.load(trained_models[2]))

    # Evaluate the trained model
    d_model.eval()
    g_model.eval()
    # e_model.eval()
    # g_model.eval()

    n = 10

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = Variable(data).cuda(device_num)

            z = Variable(torch.randn(batch_size, latent_dim)).cuda(
                device_num)
            data_augmented = g_model(z)

            comparison = data[:n]
            for a in range(n):
                comparison = torch.cat([comparison,
                                       data_augmented.view(data.size(0), 1,
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
        plt.plot(range(n_epoch), train_loss_g, '-.', label='loss_a')
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

