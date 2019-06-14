# GAN Metrics

This repository provides the code for octave conlution in GAN and evaluation. \
Evalution code is modified from https://github.com/xuqiantong/GAN-Metrics \
Original DCGAN code is from https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/dcgan/dcgan.py \
We implemented octave convolution in DCGAN and modified evaluation on different GANs.

Requirement
------

- Python 3.6.4
- torch 0.4.0
- torchvision 0.2.1
- pot 0.4.0
- tqdm 4.19.6
- numpy, scipy, math

Usage
------
dcgan.py,dcgan_octconv_gd, etc. define the generator and discriminator of GAN.
gan_eval.py calculate evaluation metrics of GAN while training it. 

```

run:
($model_name$=dcgan,dcgan_octconv_gd,dcgan_octconv_g,etc.)
'''
python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel $model_name$ \
--outf output_$model_name$ \
--sampleSize 1000
'''

examples:
python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dcgan_octconv_g \
--outf output_dcgan_octconv_g \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dcgan_octconv_d \
--outf output_dcgan_octconv_d \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dcgan_octconv_gd \
--outf output_dcgan_octconv_gd \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dragan \
--outf output_dragan \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dragan_gd \
--outf output_dragan_gd \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dcgan_g_8_2 \
--outf output_dcgan_g_8_2 \
--sampleSize 1000

run wgan:
python3 wgan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 200 \
--inputmodel dcgan \
--outf wgan_dcgan \
--sampleSize 1000
