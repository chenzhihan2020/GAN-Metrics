from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable


import metric
from metric import make_dataset
import numpy as np

## import GAN model
import dcgan_octconv_gd as inputmodel



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=32, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='results', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')

    ########################################################
    #### For evaluation ####
    parser.add_argument('--sampleSize', type=int, default=2000, help='number of samples for evaluation')
    ########################################################

    opt = parser.parse_args()
    print(opt)

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    #########################
    #### Dataset prepare ####
    #########################
    dataset = make_dataset(dataset=opt.dataset, dataroot=opt.dataroot, imageSize=opt.imageSize)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                             shuffle=True, num_workers=int(opt.workers))

    #########################
    #### Models building ####
    #########################
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ##device = torch.device("cpu")
    ngpu = int(opt.ngpu)
    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3

    netG = inputmodel.Generator().to(device)
    netG.apply(inputmodel.weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = inputmodel.Discriminator().to(device)
    netD.apply(inputmodel.weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()
    Tensor=torch.FloatTensor
    z = Variable(Tensor(np.random.normal(0, 1, (opt.batchSize, opt.nz))).to(device))
    fixed_noise=z
#   fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    # [emd-mmd-knn(knn,real,fake,precision,recall)]*4 - IS - mode_score - FID
    score_tr = np.zeros((opt.niter, 4*7+3))

    # compute initial score
    s = metric.compute_score_raw(opt.dataset, opt.imageSize, opt.dataroot, opt.sampleSize, 16, opt.outf+'/real/', opt.outf+'/fake/',
                                 netG, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
    score_tr[0] = s
    np.save('%s/score_tr.npy' % (opt.outf), score_tr)

    #########################
    #### Models training ####
    #########################
    for epoch in range(opt.niter):
        for i, data in enumerate(dataloader, 0):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            output = netD(real_cpu)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with fake
            noise = Tensor(np.random.normal(0 , 1, (opt.batchSize, opt.nz))).to(device)
            #noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            output = netD(fake)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.niter, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.detach(),
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)

        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

        ################################################
        #### metric scores computing (key function) ####
        ################################################
        s = metric.compute_score_raw(opt.dataset, opt.imageSize, opt.dataroot, opt.sampleSize, opt.batchSize, opt.outf+'/real/', opt.outf+'/fake/',\
                                     netG, opt.nz, conv_model='inception_v3', workers=int(opt.workers))
        score_tr[epoch] = s

    # save final metric scores of all epoches
    np.save('%s/score_tr_ep.npy' % opt.outf, score_tr)
    print('##### training completed :) #####')
    print('### metric scores output is scored at %s/score_tr_ep.npy ###' % opt.outf)
