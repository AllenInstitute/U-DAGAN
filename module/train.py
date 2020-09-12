import torch.optim as optim
import matplotlib.pyplot as plt
import time, os
from module.udagan import *
from module.dataloader import get_data
from module.utils import *


def train_udagan(parameters, device):
    dataloader, _ = get_data(dataset=parameters['dataset'],
                             batch_size=parameters['batch_size'],
                             file=parameters['dataset_file'],
                             n_feature=parameters['n_features'])

    saving_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    saving_path += '/results/augmenter'

    if parameters['dataset'] == 'MNIST':
        netA = Augmenter_mnist(noise_dim=parameters['num_n'],
                         latent_dim=parameters['num_z']).to(device)
        netD = Discriminator_mnist().to(device)

        saving_path += '/mnist/'


    elif parameters['dataset'] == 'FACS':
        netA = Augmenter_facs(noise_dim=parameters['num_n'],
                              latent_dim=parameters['num_z'],
                              input_dim=parameters['n_features']).to(device)
        netA.apply(weights_init)
        netD = Discriminator_facs(input_dim=parameters['n_features']).to(device)
        netD.apply(weights_init)

        saving_path = '/scRNA/'

    if parameters['initial_w']:
        netA.apply(weights_init)
        netD.apply(weights_init)

    # Loss functions
    criterionD = nn.BCELoss()
    mseDist = nn.MSELoss()

    # Set Adam optimiser for discriminator and augmenter
    optimD = optim.Adam([{'params': netD.parameters()}],
                        lr=parameters['learning_rate'])
    optimA = optim.Adam([{'params': netA.parameters()}],
                        lr=parameters['learning_rate'])

    real_label = 1
    fake_label = 0
    A_losses = []
    D_losses = []

    print('-'*50)
    print('Starting the training ...')

    for epoch in range(parameters['num_epochs']):
        epoch_start_time = time.time()
        A_loss_e = 0
        D_loss_e = 0
        recon_loss_e = 0

        if parameters['dataset'] == 'MNIST':
            for i, (data, _) in enumerate(dataloader, 0):
                b_size = parameters['batch_size']
                real_data = data.to(device)

                # Updating the discriminator -----------------------------------
                optimD.zero_grad()
                # Original samples
                label = torch.full((b_size, ), real_label, device=device)
                _, probs_real = netD(real_data)
                loss_real = criterionD(probs_real.view(-1), label)
                loss_real.backward()

                # Augmented samples
                label.fill_(fake_label)
                noise = torch.randn(b_size, parameters['num_n'], device=device)
                noise += 0.1 * torch.sign(noise)
                qz1, fake_data1 = netA(real_data, noise)
                zeros = torch.zeros(b_size, parameters['num_n'], device=device)
                qz2, fake_data2 = netA(real_data, zeros)
                _, probs_fake1 = netD(fake_data1.detach())
                _, probs_fake2 = netD(fake_data2.detach())
                loss_fake = (criterionD(probs_fake1.view(-1), label) + \
                            criterionD(probs_fake2.view(-1), label)) / 2
                loss_fake.backward()
                # Loss value for the discriminator
                D_loss = loss_real + loss_fake
                optimD.step()

                # Updating the augmenter ---------------------------------------
                optimA.zero_grad()
                # Augmented data treated as real data
                z1, probs_fake1 = netD(fake_data1)
                z2, probs_fake2 = netD(fake_data2)
                # z0, _ = netD(real_data)
                label.fill_(real_label)
                gen_loss = (criterionD(probs_fake1.view(-1), label) + \
                           criterionD(probs_fake2.view(-1), label)) / 2
                triplet_loss = TripletLoss(real_data.view(b_size, -1),
                                           fake_data2.view(b_size, -1),
                                           fake_data1.view(b_size, -1),
                                           parameters['alpha'], 'BCE')
                recon_loss = criterionD(fake_data2.view(b_size, -1),
                                real_data.view(b_size, -1))
                # Loss value for the augmenter
                A_loss = parameters['lambda'][0] * gen_loss + \
                         parameters['lambda'][1] * triplet_loss + \
                         parameters['lambda'][2] * KL_dist(qz1, qz2) + \
                         parameters['lambda'][3] * recon_loss

                # parameters['lambda'][2] * mseDist(qz1, qz2)

                A_loss.backward()
                optimA.step()

                A_losses.append(A_loss.data.item())
                D_losses.append(D_loss.data.item())
                A_loss_e += A_loss.data.item()
                D_loss_e += D_loss.data.item()
                recon_loss_e += recon_loss.data.item()

        else:
            for i, (data, data_bin, _) in enumerate(dataloader, 0):
                b_size = parameters['batch_size']
                real_data = data.to(device)
                real_data_bin = data_bin.to(device)

                # Updating the discriminator -----------------------------------
                optimD.zero_grad()
                # Original samples
                label = torch.full((b_size,), real_label, device=device)
                _, probs_real = netD(real_data_bin)
                loss_real = criterionD(probs_real.view(-1), label)

                if F.relu(loss_real - 0.25) > 0:
                    loss_r = loss_real
                else:
                    loss_r = F.relu(loss_real - 0.25)

                loss_r.backward()

                # Augmented samples
                label.fill_(fake_label)
                noise = torch.randn(b_size, parameters['num_n'], device=device)
                noise += 0.1 * torch.sign(noise)
                qz1, fake_data1 = netA(real_data_bin, noise)
                zeros = torch.zeros(b_size, parameters['num_n'], device=device)
                qz2, fake_data2 = netA(real_data_bin, zeros)
                # binarizing the augmented sample
                fake_data1_bin = fake_data1.clone()
                fake_data2_bin = fake_data2.clone()
                fake_data1_bin[fake_data1_bin > 0] = 1.
                fake_data2_bin[fake_data2_bin > 0] = 1.
                _, probs_fake1 = netD(fake_data1_bin.detach())
                _, probs_fake2 = netD(fake_data2_bin.detach())
                loss_fake = (criterionD(probs_fake1.view(-1), label) + \
                             criterionD(probs_fake2.view(-1), label)) / 2

                if F.relu(loss_fake - 0.25) > 0:
                    loss_f = loss_fake
                else:
                    loss_f = F.relu(loss_fake - 0.25)

                loss_f.backward()
                # Loss value for the discriminator
                D_loss = loss_real + loss_fake
                optimD.step()

                # Updating the augmenter ---------------------------------------
                optimA.zero_grad()
                # Augmented data treated as real data
                z1, probs_fake1 = netD(fake_data1_bin)
                z2, probs_fake2 = netD(fake_data2_bin)
                # z0, _ = netD(real_data)
                label.fill_(real_label)
                gen_loss = (criterionD(probs_fake1.view(-1), label) + \
                            criterionD(probs_fake2.view(-1), label)) / 2
                triplet_loss = TripletLoss(real_data_bin.view(b_size, -1),
                                           fake_data2_bin.view(b_size, -1),
                                           fake_data1_bin.view(b_size, -1),
                                           parameters['alpha'], 'BCE')
                # recon_loss = mseDist(fake_data2.view(b_size, -1),
                #                      real_data.view(b_size, -1))
                recon_loss = F.mse_loss(fake_data2, real_data, reduction='mean')
                # Loss value for the augmenter
                A_loss = parameters['lambda'][0] * gen_loss + \
                         parameters['lambda'][1] * triplet_loss + \
                         parameters['lambda'][2] * mseDist(qz1, qz2) + \
                         parameters['lambda'][3] * recon_loss
                A_loss.backward()
                optimA.step()

                A_losses.append(A_loss.data.item())
                D_losses.append(D_loss.data.item())
                A_loss_e += A_loss.data.item()
                D_loss_e += D_loss.data.item()
                recon_loss_e += recon_loss.data.item()

        A_loss_epoch = A_loss_e / (i + 1)
        D_loss_epoch = D_loss_e / (i + 1)
        recon_loss_epoch = recon_loss_e / (i + 1)

        print('=====> Epoch:{}, Augmenter Loss: {:.4f}, Discriminator Loss: {'
              ':.4f}, Recon Loss: {:.4f}, Elapsed Time:{:.2f}'.format(epoch,
                A_loss_epoch,
                D_loss_epoch, recon_loss_epoch, time.time() - epoch_start_time))

    print("-" * 50)
    # Save trained models
    if parameters['save']:
        torch.save({
            'netA': netA.state_dict(),
            'netD': netD.state_dict(),
            'optimD': optimD.state_dict(),
            'optimA': optimA.state_dict(),
            'parameters': parameters
            }, saving_path + 'model_bs_%d_dn_%d_dz_%d_lambda'
               %(parameters['batch_size'], parameters['num_n'], parameters[
            'num_z']))

        # Plot the training losses.
        plt.figure()
        plt.title("Augmenter and Discriminator Loss Values in Training")
        plt.plot(A_losses, label="A")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(saving_path + 'loss_curve.png')



def train_vaegan(parameters, device):

    dataloader, _ = get_data(dataset=parameters['dataset'],
                              batch_size=parameters['batch_size'],
                              file=parameters['dataset_file'],
                              n_feature=parameters['n_features'],
                              training=True,
                              remove_nonneuron=parameters['remove_nonneuron'],
                              remove_CR_Meis2=parameters['remove_CR_Meis2'])

    saving_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
    saving_path += '/results/generator'

    if parameters['dataset'] == 'MNIST':
        netG = Generator_mnist(latent_dim=parameters['num_z']).to(device)
        netD = Discriminator_mnist().to(device)

        saving_path += '/mnist/'


    elif parameters['dataset'] == 'FACS':
        netG = Generator_facs(latent_dim=parameters['num_z'],
                              input_dim=parameters['n_features']).to(device)
        netG.apply(weights_init)
        netD = Discriminator_facs(input_dim=parameters['n_features']).to(device)
        netD.apply(weights_init)

        saving_path += '/scRNA/'

    if parameters['initial_w']:
        netG.apply(weights_init)
        netD.apply(weights_init)

    # Loss functions
    criterionD = nn.BCELoss()
    criterionQ_con = NormalNLLLoss()
    mseDist = nn.MSELoss()

    # Set Adam optimiser for discriminator and augmenter
    optimD = optim.Adam([{'params': netD.parameters()}],
                        lr=parameters['learning_rate'])
    optimG = optim.Adam([{'params': netG.parameters()}],
                        lr=parameters['learning_rate'])

    real_label = 1
    fake_label = 0
    G_losses = []
    D_losses = []

    print('-'*50)
    print('Starting the training ...')

    for epoch in range(parameters['num_epochs']):
        epoch_start_time = time.time()
        G_loss_e = 0
        D_loss_e = 0
        recon_loss_e = 0
        gen_loss_e = 0

        if parameters['dataset'] == 'MNIST':
            for i, (data, _) in enumerate(dataloader, 0):
                b_size = parameters['batch_size']
                real_data = data.to(device)

                # Updating the discriminator -----------------------------------
                optimD.zero_grad()
                # Original samples
                label = torch.full((b_size, ), real_label, device=device)
                _, probs_real = netD(real_data)
                loss_real = criterionD(probs_real.view(-1), label)
                loss_real.backward()

                # Augmented samples
                label.fill_(fake_label)
                _, fake_data = netG(real_data)
                zeros = torch.zeros(b_size, parameters['num_n'], device=device)
                _, probs_fake = netD(fake_data.detach())
                loss_fake = criterionD(probs_fake.view(-1), label)
                loss_fake.backward()
                # Loss value for the discriminator
                D_loss = loss_real + loss_fake
                optimD.step()

                # Updating the augmenter ---------------------------------------
                optimG.zero_grad()
                # Augmented data treated as real data
                z, probs_fake = netD(fake_data)
                label.fill_(real_label)
                gen_loss = criterionD(probs_fake.view(-1), label)
                recon_loss = criterionD(fake_data.view(b_size, -1),
                                real_data.view(b_size, -1))
                # Loss value for the augmenter
                G_loss = parameters['lambda'][0] * gen_loss + \
                         parameters['lambda'][3] * recon_loss
                G_loss.backward()
                optimG.step()

                G_losses.append(G_loss.data.item())
                D_losses.append(D_loss.data.item())
                G_loss_e += G_loss.data.item()
                D_loss_e += D_loss.data.item()
                recon_loss_e += recon_loss.data.item()
                gen_loss_e += gen_loss.data.item()

        else:
            for i, (data, data_bin, _) in enumerate(dataloader, 0):
                b_size = parameters['batch_size']
                real_data = data.to(device)
                real_data_bin = data_bin.to(device)

                # Updating the discriminator -----------------------------------
                optimD.zero_grad()
                # Original samples
                label = torch.full((b_size,), real_label, device=device)
                _, probs_real = netD(real_data_bin)
                loss_real = criterionD(probs_real.view(-1), label)

                if F.relu(loss_real - np.log(2)/2) > 0:
                    loss_real.backward()
                    optim_D = True
                else:
                    optim_D = False

                # Augmented samples
                label.fill_(fake_label)
                _, fake_data = netG(real_data_bin)
                # binarizing the augmented sample
                fake_data_bin = fake_data.clone()
                fake_data_bin[fake_data_bin > 0] = 1.
                _, probs_fake = netD(fake_data_bin.detach())
                loss_fake = criterionD(probs_fake.view(-1), label)

                if F.relu(loss_fake - np.log(2)/2) > 0:
                    loss_fake.backward()
                    optim_D = True

                # Loss value for the discriminator
                D_loss = loss_real + loss_fake

                if optim_D:
                    optimD.step()

                # Updating the augmenter ---------------------------------------
                optimG.zero_grad()
                # Augmented data treated as real data
                _, probs_fake = netD(fake_data_bin)
                label.fill_(real_label)
                gen_loss = criterionD(probs_fake.view(-1), label)
                recon_loss = F.mse_loss(fake_data, real_data, reduction='mean')
                # Loss value for the augmenter
                G_loss = parameters['lambda'][0] * gen_loss + \
                         parameters['lambda'][3] * recon_loss
                G_loss.backward()
                optimG.step()

                G_losses.append(G_loss.data.item())
                D_losses.append(D_loss.data.item())
                G_loss_e += G_loss.data.item()
                D_loss_e += D_loss.data.item()
                recon_loss_e += recon_loss.data.item()
                gen_loss_e += gen_loss.data.item()

        G_loss_epoch = G_loss_e / (i + 1)
        D_loss_epoch = D_loss_e / (i + 1)
        recon_loss_epoch = recon_loss_e / (i + 1)
        gen_loss_epoch = gen_loss_e / (i + 1)

        print('=====> Epoch:{}, Generator Loss: {:.4f}, Discriminator Loss: {'
              ':.4f}, Recon Loss: {:.4f}, Gen Loss: {:.4f}, Elapsed Time:{'
              ':.2f}'.format(epoch, G_loss_epoch, D_loss_epoch,
            recon_loss_epoch, gen_loss_epoch, time.time() - epoch_start_time))

    print("-" * 50)
    # Save trained models
    if parameters['save']:
        torch.save({
            'netG': netG.state_dict(),
            'netD': netD.state_dict(),
            'optimD': optimD.state_dict(),
            'optimG': optimG.state_dict(),
            'parameters': parameters
            }, saving_path + 'modelG_bs_%d_dn_%d_dz_%d'
               %(parameters['batch_size'], parameters['num_z'], parameters[
            'num_epochs']))

        # Plot the training losses.
        plt.figure()
        plt.title("Generator and Discriminator Loss Values in Training")
        plt.plot(G_losses, label="G")
        plt.plot(D_losses, label="D")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(saving_path + 'loss_curve_g.png')