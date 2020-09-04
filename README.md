# Unsupervised Adversarial Augmenter/Generator Networks
This repository contains an implementation of an unspervised adverserial framework for both data augmentation and generarion.

## Table of contents
* [Generator](#generator)
* [Augmenter](#augmenter)
* [Usage](#usage)

## Generator
combining a variational autoencoder (VAE) with a generative adversarial network (GAN), we introduce a VAE-GAN netowrk that leverages unsupervised representation learning and data sample reconstruction.

	
## Augmenter
Given the adversarial training proposed for GANs, here we introduce an augmentation network thatgenerates multiple nonidentical augmented samples with identical class labels, called U-DAGAN.

The schematic of the proposed architecture for unsupervised data augmentation and the augmenter's architecture.
<img src="figures/udagan.png" width="400">
<img src="figures/augmenter.png" width="300">

### Example
#### MNIST
<img src="figures/mnist_augmented_sample.png" width="800">
#### snRNA-seq (FACS)
<img src="figures/sc_lowD_feature_space.png" width="800">

<img src="figures/sc_gene_82.png" width="800">

## Usage
Each **generator** and **augmenter** folder contains code for training the network and generating fakes samples.
To train the model you can use `run.py` to train the network and `*augmenter.py`/`*augmenter.py` to generate fake samples.

