# GAN Metrics

This repository provides the code for [An empirical study on evaluation metrics of generative adversarial networks](https://arxiv.org/abs/1806.07755).

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

- We create a demo for DCGAN training as well as computing all the metrics after each epoch.     
In the demo, final metrics scores of all epoches will be scored in `<outf>/score_tr_ep.npy`    
- If you want to compute metrics of your own images, you have to modify the codes of function `compute_score_raw()` in `metric.py` by yourself :)

```

run model:
($model_name$=dcgan,dcgan_octconv_gd,dcgan_octconv_g,dcgan_octconv_d)
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
--niter 100 \
--inputmodel dcgan_octconv_g \
--outf output_dcgan_octconv_g \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 100 \
--inputmodel dcgan_octconv_d \
--outf output_dcgan_octconv_d \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 100 \
--inputmodel dcgan_octconv_gd \
--outf output_dcgan_octconv_gd \
--sampleSize 1000

python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 100 \
--inputmodel dragan \
--outf output_dragan \
--sampleSize 1000

examples:
python3 gan_eval.py \
--dataset cifar10 \
--dataroot input_data \
--cuda \
--batchSize 64 \
--niter 100 \
--inputmodel dragan_gd \
--outf output_dragan_gd \
--sampleSize 1000
