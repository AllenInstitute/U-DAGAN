import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

eps = 1e-6

def weights_init(m):
    """
    Initialise weights of the networks
    """
    if(type(m) == nn.ConvTranspose2d or type(m) == nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.1)
    elif(type(m) == nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, 1.0, 0.1)
        nn.init.constant_(m.bias.data, 0)


class NormalNLLLoss:
    """
    Calculate the negative log likelihood
    of normal distribution.
    This needs to be minimised

    Treating Q(cj | x) as a factored Gaussian
    """
    def __call__(self, x, mu, var):
        
        logli = -0.5 * (var.mul(2 * np.pi) + eps).log() - (x - mu).pow(2).div(
            var.mul(2.0) + eps)
        nll = -(logli.sum(1).mean())
        return nll


def TripletLoss(anchor, positive, negative, margin=0.2, loss='BCE'):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    if loss == 'BCE':
        dist = nn.BCELoss()
    elif loss == 'MSE':
        dist = nn.MSELoss()

    distance_positive = dist(positive, anchor)
    distance_negative = dist(negative, anchor)
    losses = F.relu(distance_positive - distance_negative + margin)

    return losses.mean()


def reparam_trick(mu, std):
    """
    Generate samples from a normal distribution for reparametrization trick.

    input args
        mu: mean of the Gaussian distribution for q(s|z,x) = N(mu, sigma^2*I).
        log_sigma: log of variance of the Gaussian distribution for
                   q(s|z,x) = N(mu, sigma^2*I).

    return
        a sample from Gaussian distribution N(mu, sigma^2*I).
    """
    eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
    return eps.mul(std).add(mu)
