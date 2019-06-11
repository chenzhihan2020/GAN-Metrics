import argparse
import os
import numpy as np
#import math
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image

#from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.autograd as autograd

import torch.nn as nn
#import torch.nn.functional as F
import torch

latent_dim=100
channels=3
img_size=32


def weights_init(m):
    classname = m.__class__.__name__
    
    if classname.find("OctaveConv") != -1:
        if m.conv_l2l is not None:
            torch.nn.init.normal_(m.conv_l2l.weight.data, 0.0, 0.02)
        if m.conv_h2l is not None:
            torch.nn.init.normal_(m.conv_h2l.weight.data, 0.0, 0.02)
        if m.conv_l2h is not None:
            torch.nn.init.normal_(m.conv_l2h.weight.data, 0.0, 0.02)
        if m.conv_h2h is not None:
            torch.nn.init.normal_(m.conv_h2h.weight.data, 0.0, 0.02)
    elif classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class OctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False):
        super(OctaveConv, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride
        assert 0 <= alpha_in <= 1 and 0 <= alpha_out <= 1, "Alphas should be in the interval from 0 to 1."
        self.alpha_in, self.alpha_out = alpha_in, alpha_out
        self.conv_l2l = None if alpha_in == 0 or alpha_out == 0 else \
                        nn.Conv2d(int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_l2h = None if alpha_in == 0 or alpha_out == 1 else \
                        nn.Conv2d(int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2l = None if alpha_in == 1 or alpha_out == 0 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)
        self.conv_h2h = None if alpha_in == 1 or alpha_out == 1 else \
                        nn.Conv2d(in_channels - int(alpha_in * in_channels), out_channels - int(alpha_out * out_channels),
                                  kernel_size, 1, padding, dilation, groups, bias)

    def forward(self, x):
        x_h, x_l = x if type(x) is tuple else (x, None)

        if x_h is not None:
            x_h = self.downsample(x_h) if self.stride == 2 else x_h
            x_h2h = self.conv_h2h(x_h)
            x_h2l = self.conv_h2l(self.downsample(x_h)) if self.alpha_out > 0 else None
        if x_l is not None:
            x_l2h = self.conv_l2h(x_l)
            x_l2h = self.upsample(x_l2h) if self.stride == 1 else x_l2h
            x_l2l = self.downsample(x_l) if self.stride == 2 else x_l
            x_l2l = self.conv_l2l(x_l2l) if self.alpha_out > 0 else None 
            x_h = x_l2h + x_h2h
            x_l = x_h2l + x_l2l if x_h2l is not None and x_l2l is not None else None
            return x_h, x_l
        else:
            return x_h2h, x_h2l


class Conv_BN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d):
        super(Conv_BN, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.bn_h(x_h)
        x_l = self.bn_l(x_l) if x_l is not None else None
        return x_h, x_l


class Conv_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(Conv_BN_ACT, self).__init__()
        self.conv = OctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding, dilation,
                               groups, bias)
        self.bn_h = None if alpha_out == 1 else norm_layer(int(out_channels * (1 - alpha_out)))
        self.bn_l = None if alpha_out == 0 else norm_layer(int(out_channels * alpha_out))
        self.act = activation_layer(inplace=True)

    def forward(self, x):
        x_h, x_l = self.conv(x)
        x_h = self.act(self.bn_h(x_h))
        x_l = self.act(self.bn_l(x_l)) if x_l is not None else None
        return x_h, x_l



class dual_channel_upsample(nn.Module):
    def __init__(self, scale_factor=2):
        super(dual_channel_upsample, self).__init__()
        self.upsampler = nn.Upsample(scale_factor=scale_factor)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.upsampler(x_h)
        x_l = self.upsampler(x_l)
        return x_h, x_l
    
class dual_channel_batchnorm2d(nn.Module):
    def __init__(self, size, eps, alpha=0.5):
        super(dual_channel_batchnorm2d, self).__init__()
        self.batcher_h = nn.BatchNorm2d(size - int(size*alpha), eps)
        self.batcher_l = nn.BatchNorm2d(int(size*alpha), eps)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher_h(x_h)
        if x_l is not None:
            x_l = self.batcher_l(x_l)
        return x_h, x_l
    
class dual_channel_leakyrelu(nn.Module):
    def __init__(self, factor, inplace=True):
        super(dual_channel_leakyrelu, self).__init__()
        self.batcher = nn.LeakyReLU(factor)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        if x_l is not None:
            x_l = self.batcher(x_l)
        return x_h, x_l
    
class high_channel_tanh(nn.Module):
    def __init__(self):
        super(high_channel_tanh, self).__init__()
        self.batcher = nn.Tanh()
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        return x_h

class dual_channel_dropout(nn.Module):
    def __init__(self, p, inplace=False):
        super(dual_channel_dropout, self).__init__()
        self.batcher = nn.Dropout2d(p)
    def forward(self, x):
        x_h, x_l = x
        x_h = self.batcher(x_h)
        if x_l is not None:
            x_l = self.batcher(x_l) 
        return x_h, x_l

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            OctaveConv(128, 128, 3, stride=1, padding=1, alpha_in = 0),
            dual_channel_batchnorm2d(128, 0.8),
            dual_channel_leakyrelu(0.2, inplace=True),
            dual_channel_upsample(scale_factor=2),
            OctaveConv(128, 64, 3, stride=1, padding=1),
            dual_channel_batchnorm2d(64, 0.8),
            dual_channel_leakyrelu(0.2, inplace=True),
            OctaveConv(64, channels, 3, stride=1, padding=1, alpha_out = 0),
            high_channel_tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, alpha_in=0.5, alpha_out = 0.5, bn=True):
            block = [OctaveConv(in_filters, out_filters, 3, alpha_in, alpha_out, stride=2, padding=1), 
                     dual_channel_leakyrelu(0.2, inplace=True), 
                     dual_channel_dropout(0.25, inplace=False)]
            if bn:
                block.append(dual_channel_batchnorm2d(out_filters, 0.8, alpha_out))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, alpha_in=0, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128, alpha_out=0),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out, _ = out
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Loss weight for gradient penalty
lambda_gp = 10


Tensor = torch.cuda.FloatTensor


def compute_gradient_penalty(D, X):
    """Calculates the gradient penalty loss for DRAGAN"""
    # Random weight term for interpolation
    alpha = Tensor(np.random.random(size=X.shape))

    interpolates = alpha * X + ((1 - alpha) * (X + 0.5 * X.std() * torch.rand(X.size())))
    interpolates = Variable(interpolates, requires_grad=True)

    d_interpolates = D(interpolates)

    fake = Variable(Tensor(X.shape[0], 1).fill_(1.0), requires_grad=False)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

