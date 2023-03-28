# Spectral Hint GAN

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/FcF-Inpainting/blob/main/colab/FcF_Inpainting.ipynb) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/shi-lab/FcF-Inpainting)  -->
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repo hosts the official implementary of:

[Xingqian Xu](https://ifp-uiuc.github.io/), Shant Navasardyan, Vahram Tadevosyan, Andranik Sargsyan, Yadong Mu and [Humphrey Shi](https://www.humphreyshi.com/home), **Image Completion with Heterogeneously Filtered Spectral Hints**, [Paper arXiv Link](https://arxiv.org/abs/2211.03700).

## News

- [2022.11.12]: Evaluation code and pretrained model released.
- [2022.11.07]: Our paper is accepted in WACV23.
- [2022.11.06]: Repo initiated.

## Introduction

<p align="center">
  <img src="assets/teaser.png" width="99%">
</p>

Spectral Hint GAN (**SH-GAN**) is an high-performing inpainting network enpowered by CoModGAN and novel spectral processing techniques. SH-GAN reaches state-of-the-art on FFHQ and Places2 with freeform masks.

## Network and Algorithm

The overall structure of our SH-GAN shows in the following figure:

<p align="center">
  <img src="assets/network.png" width="99%">
</p>

The sturcture of our Spectral Hint Unit shows in the following graph:

<p align="center">
  <img src="assets/shu.png" width="40%">
</p>

Heterogeneous Filtering Explaination: 
* 1x1 Convolution in Fourier domain leads a uniform (homogeneous) transform from one spectral space to another.
* ReLU in Fourier domain is like a value-dependend band pass filter that zero out some frequency values.
* We promote the __heterogeneous transforms__ in spectral space, in which the frequency value transformations are depended on the frequency bands.

<p align="center">
  <img src="assets/hfilter.png" width="80%">
</p>

Gaussian Split Algorithm Explaination:

* Gaussian Split is a spectral space downsampling method that well-suit deep learning structures. A quick intuition is that it likes Wavelet Transform that can pass information in different frequency band to its corresponding resolution.

<p align="center">
  <img src="assets/split.png" width="99%">
</p>

## Data

We use FFHQ and Places2 as our main dataset. Download these dataset from the following official link: [FFHQ](https://github.com/NVlabs/ffhq-dataset), [Places2](http://places2.csail.mit.edu/)

Directory of FFHQ data for our code:

```
├── data
│   └── ffhq
│       └── ffhq256x256.zip
│       └── ffhq512x512.zip
```

Directory of Places2 data for our code:

* Download the data_challenge.zip from Places2 official website and decompress it to /data/Places2
* Same for val_large.zip

```
├── data
│   └── Places2
│       └── data_challenge
│           ...
│       └── val_large
│           ...
```

## Setup

```
conda create -n shgan python=3.8
conda activate shgan
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install -r requirement.txt
```

## Results and pretrained models

|   |DIM|DATA|FID|LPIPS|PSNR|SSIM|Download|
|---|---|---|---|---|---|---|---|
|CoModGAN|256|FFHQ   |4.7755|0.2568|16.24|0.5913||
|SH-GAN  |256|FFHQ   |4.3459|0.2542|16.37|0.5911|[link](https://drive.google.com/file/d/1XYUAA5OF1PH-ANi6TzpeVjswxF-yQZYz/view?usp=sharing)|
|CoModGAN|512|FFHQ   |3.6996|0.2469|18.46|0.6956||
|SH-GAN  |512|FFHQ   |3.4134|0.2447|18.43|0.6936|[link](https://drive.google.com/file/d/1wtW-nEGu_8cka7WmZ0ctxIdvr26aNFJu/view?usp=sharing)|
|CoModGAN|256|Places2|9.3621|0.3990|14.50|0.4923||
|SH-GAN  |256|Places2|7.5036|0.3940|14.58|0.4958|[link](https://drive.google.com/file/d/1Kw2u_9R7IVUd_W7zpEASKBa5-oX4tCqN/view?usp=sharing)|
|CoModGAN|512|Places2|7.9735|0.3420|16.00|0.5953||
|SH-GAN  |512|Places2|7.0277|0.3386|16.03|0.5973|[link](https://drive.google.com/file/d/1MKQF266xZ3wJ6wJ6J8S6zW_iaOJWLsND/view?usp=sharing)|

## Evaluation

Here are the one-line shell commends to evaluation SH-GAN on FFHQ 256/512 and Places2 256/512.

```
python main.py --experiment shgan_ffhq256_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --experiment shgan_ffhq512_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --experiment shgan_places256_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --experiment shgan_places512_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
```

Also you need to:
* Download the data, put them as the directories mentioned in Data session.
* Create ```./pretrained``` and move all downloaded pretrained models in it.
* Create ```./log/shgan_ffhq/99999_eval``` and ```./log/shgan_places2/99999_eval```

Some simple things to do to resolve the issues:
* The evaluation code caches and later relys on ```.cache/****_real_feat.npy``` for FID calculation. If it corrupts, numbers will be wrong. But you can simple remove it and the code will auto recompute one.
* The final stage of FID computation requires CPU resource so it is normal to be slow, so be patient.


## Training

coming soon

## Citation

```
@inproceedings{xu2023image,
  title={Image Completion with Heterogeneously Filtered Spectral Hints},
  author={Xu, Xingqian and Navasardyan, Shant and Tadevosyan, Vahram and Sargsyan, Andranik and Mu, Yadong and Shi, Humphrey},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={4591--4601},
  year={2023}
}
```

## Acknowledgement

Part of the codes reorganizes/reimplements code from the following repositories: [Comodgan official Github](https://github.com/zsyzzsoft/co-mod-gan) and [Stylegan2-ADA official Github](https://github.com/NVlabs/stylegan2-ada-pytorch/).
