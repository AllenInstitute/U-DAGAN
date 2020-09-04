import torch
import torch.nn as nn
import torch.nn.functional as F
from module.utils import *


class Augmenter_mnist(nn.Module):
    def __init__(self, noise_dim, latent_dim):
        super().__init__()

        self.noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.bnz = nn.BatchNorm1d(self.noise.out_features)

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(self.conv1.out_channels,
                               self.conv1.out_channels * 2, 4, 2, 1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(self.conv2.out_channels, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(self.conv3.out_channels + noise_dim,
                               (self.conv3.out_channels + noise_dim) // 8, 1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.conv_s = nn.Conv2d(self.conv4.out_channels, latent_dim, 1,
                                bias=False)
        self.bn_s = nn.BatchNorm2d(self.conv_s.out_channels)

        self.tconv0 = nn.ConvTranspose2d(latent_dim, 128, 1, 1, bias=False)
        self.tbn0 = nn.BatchNorm2d(self.tconv0.out_channels)

        self.tconv1 = nn.ConvTranspose2d(self.tconv0.out_channels, 1024, 1,
                                         1, bias=False)
        self.tbn1 = nn.BatchNorm2d(self.tconv1.out_channels)

        self.tconv2 = nn.ConvTranspose2d(self.tconv1.out_channels,
                                         self.tconv1.out_channels // 8, 7, 1,
                                         bias=False)
        self.tbn2 = nn.BatchNorm2d(self.tconv2.out_channels)

        self.tconv3 = nn.ConvTranspose2d(self.tconv2.out_channels,
                                         self.tconv2.out_channels // 2, 4, 2, \
                                               padding=1, bias=False)
        self.tbn3 = nn.BatchNorm2d(self.tconv3.out_channels)

        self.tconv4 = nn.ConvTranspose2d(self.tconv3.out_channels, 1, 4, 2,
                                         padding=1, bias=False)

    def forward(self, x, z):

        z = F.elu(self.bnz(self.noise(z)))
        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = torch.cat((x.squeeze(), z), dim=1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
        s = self.bn_s(self.conv_s(x))
        x = F.relu(self.tbn0(self.tconv0(s)))
        x = F.relu(self.tbn1(self.tconv1(x)))
        x = F.relu(self.tbn2(self.tconv2(x)))
        x = F.relu(self.tbn3(self.tconv3(x)))

        return s.squeeze(), torch.sigmoid(self.tconv4(x))

class Generator_mnist(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(self.conv1.out_channels,
                               self.conv1.out_channels * 2, 4, 2, 1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)

        self.conv3 = nn.Conv2d(self.conv2.out_channels, 1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)

        self.conv4 = nn.Conv2d(self.conv3.out_channels,
                               self.conv3.out_channels // 8, 1,
                               bias=False)
        self.bn4 = nn.BatchNorm2d(self.conv4.out_channels)

        self.fc_mu = nn.Linear(self.conv4.out_channels, latent_dim)
        self.fc_sigma = nn.Linear(self.conv4.out_channels, latent_dim)
        self.bn_mu = nn.BatchNorm1d(self.fc_mu.out_features)

        self.tconv0 = nn.ConvTranspose2d(latent_dim, 128, 1, 1, bias=False)
        self.tbn0 = nn.BatchNorm2d(self.tconv0.out_channels)

        self.tconv1 = nn.ConvTranspose2d(self.tconv0.out_channels, 1024, 1,
                                         1, bias=False)
        self.tbn1 = nn.BatchNorm2d(self.tconv1.out_channels)

        self.tconv2 = nn.ConvTranspose2d(self.tconv1.out_channels,
                                         self.tconv1.out_channels // 8, 7, 1,
                                         bias=False)
        self.tbn2 = nn.BatchNorm2d(self.tconv2.out_channels)

        self.tconv3 = nn.ConvTranspose2d(self.tconv2.out_channels,
                                         self.tconv2.out_channels // 2, 4, 2, \
                                               padding=1, bias=False)
        self.tbn3 = nn.BatchNorm2d(self.tconv3.out_channels)

        self.tconv4 = nn.ConvTranspose2d(self.tconv3.out_channels, 1, 4, 2,
                                         padding=1, bias=False)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.1, inplace=True)
        mu = self.bn_mu(self.fc_mu(x))
        sigma = torch.sigmoid(self.fc_sigma(x))
        s = reparam_trick(mu, sigma)
        x = F.relu(self.tbn0(self.tconv0(s)))
        x = F.relu(self.tbn1(self.tconv1(x)))
        x = F.relu(self.tbn2(self.tconv2(x)))
        x = F.relu(self.tbn3(self.tconv3(x)))

        return s.squeeze(), torch.sigmoid(self.tconv4(x))

class Discriminator_mnist(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(self.conv1.out_channels,
                               self.conv1.out_channels * 2, 4, 2, 1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.conv3 = nn.Conv2d(self.conv2.out_channels,
                               1024, 7, bias=False)
        self.bn3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.conv = nn.Conv2d(self.conv3.out_channels, 1, 1)

    def forward(self, x):

        x = F.leaky_relu(self.conv1(x), 0.1, inplace=True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.1, inplace=True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.1, inplace=True)
        output = torch.sigmoid(self.conv(x))

        return x, output



class Augmenter_facs(nn.Module):
    def __init__(self, noise_dim, latent_dim, input_dim=5000, n_dim=500,
                 p_drop=0.5):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)

        self.noise = nn.Linear(noise_dim, noise_dim, bias=False)
        self.bnz = nn.BatchNorm1d(self.noise.out_features)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc5 = nn.Linear(n_dim + noise_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fcs = nn.Linear(self.fc5.out_features, latent_dim)
        self.batch_fcs = nn.BatchNorm1d(num_features=self.fcs.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc6 = nn.Linear(self.fcs.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc10 = nn.Linear(self.fc9.out_features, self.fc9.out_features)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x, z):

        z = F.elu(self.bnz(self.noise(z)))
        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        x = F.relu(self.batch_fc3(self.fc3(x)))
        x = F.relu(self.batch_fc4(self.fc4(x)))
        x = torch.cat((x, z), dim=1)
        x = F.relu(self.batch_fc5(self.fc5(x)))
        s = self.batch_fcs(self.fcs(x))
        x = F.relu((self.fc6(s)))
        x = F.relu((self.fc7(x)))
        x = F.relu((self.fc8(x)))
        x = F.relu((self.fc9(x)))
        x = F.relu((self.fc10(x)))

        # x = F.relu(self.batch_fc6(self.fc6(s)))
        # x = F.relu(self.batch_fc7(self.fc7(x)))
        # x = F.relu(self.batch_fc8(self.fc8(x)))
        # x = F.relu(self.batch_fc9(self.fc9(x)))
        # x = F.relu(self.batch_fc10(self.fc10(x)))

        return s, F.relu(self.fc11(x))


class Generator_facs(nn.Module):
    def __init__(self, latent_dim, input_dim=5000, n_dim=500,
                 p_drop=0.0):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc4 = nn.Linear(n_dim, n_dim)
        self.batch_fc4 = nn.BatchNorm1d(num_features=self.fc4.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc5 = nn.Linear(n_dim, n_dim // 5)
        self.batch_fc5 = nn.BatchNorm1d(num_features=self.fc5.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc_mu = nn.Linear(self.fc5.out_features, latent_dim)
        self.fc_sigma = nn.Linear(self.fc5.out_features, latent_dim)
        self.batch_fc_mu = nn.BatchNorm1d(num_features=self.fc_mu.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc6 = nn.Linear(self.fc_mu.out_features, n_dim // 5)
        self.batch_fc6 = nn.BatchNorm1d(num_features=self.fc6.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc7 = nn.Linear(self.fc6.out_features, n_dim)
        self.batch_fc7 = nn.BatchNorm1d(num_features=self.fc7.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc8 = nn.Linear(n_dim, n_dim)
        self.batch_fc8 = nn.BatchNorm1d(num_features=self.fc8.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc9 = nn.Linear(n_dim, input_dim // 5)
        self.batch_fc9 = nn.BatchNorm1d(num_features=self.fc9.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc10 = nn.Linear(self.fc9.out_features, self.fc9.out_features)
        self.batch_fc10 = nn.BatchNorm1d(num_features=self.fc10.out_features,
                                        eps=1e-10, momentum=moment,
                                         affine=False)
        self.fc11 = nn.Linear(self.fc10.out_features, input_dim)

    def forward(self, x):

        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        x = F.relu(self.batch_fc3(self.fc3(x)))
        x = F.relu(self.batch_fc4(self.fc4(x)))
        x = F.relu(self.batch_fc5(self.fc5(x)))
        mu = self.batch_fc_mu(self.fc_mu(x))
        sigma = torch.sigmoid(self.fc_sigma(x))
        s = reparam_trick(mu, sigma)
        x = F.relu((self.fc6(s)))
        x = F.relu((self.fc7(x)))
        x = F.relu((self.fc8(x)))
        x = F.relu((self.fc9(x)))
        x = F.relu((self.fc10(x)))

        return s, F.relu(self.fc11(x))



class Discriminator_facs(nn.Module):
    def __init__(self, input_dim=5000, n_dim=500, p_drop=0.0):
        super().__init__()

        moment = 0.01
        self.dp = nn.Dropout(p_drop)

        self.fc1 = nn.Linear(input_dim, input_dim // 5)
        self.batch_fc1 = nn.BatchNorm1d(num_features=self.fc1.out_features,
                                        eps=1e-10, momentum=moment,
                                        affine=False)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features)
        self.batch_fc2 = nn.BatchNorm1d(num_features=self.fc2.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.fc3 = nn.Linear(self.fc2.out_features, n_dim)
        self.batch_fc3 = nn.BatchNorm1d(num_features=self.fc3.out_features,
                                        eps=1e-10, momentum=moment, affine=False)
        self.disc = nn.Linear(self.fc2.out_features, 1, 1)

    def forward(self, x):

        x = F.relu(self.batch_fc1(self.fc1(self.dp(x))))
        x = F.relu(self.batch_fc2(self.fc2(x)))
        # x = F.relu(self.batch_fc3(self.fc3(x)))
        output = torch.sigmoid(self.disc(x))

        return x, output