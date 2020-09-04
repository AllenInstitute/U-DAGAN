# Unsupervised Adversarial Augmenter/Generator Networks
This repository contains an implementation of an unspervised adverserial framework for both data augmentation and generarion.

## Table of contents
* [Generator](#generator)
* [Augmenter](#augmenter)

## Generator
combining a variational autoencoder (VAE) with a generative adversarial network (GAN), we introduce a VAE-GAN netowrk that leverages unsupervised representation learning and data sample reconstruction.

	
## Augmenter
Given the adversarial training proposed for GANs, here we introduce an augmentation network thatgenerates multiple nonidentical augmented samples with identical class labels, called U-DAGAN.

Schematic of the proposed architecture for unsupervised data augmentation,
https://github.com/AllenInstitute/U-DAGAN/blob/master/figures/augmenter.png

and the augmenter's architecture.
https://github.com/AllenInstitute/U-DAGAN/blob/master/figures/udagan.png
