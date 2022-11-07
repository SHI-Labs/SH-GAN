# Spectral Hint GAN

<!-- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SHI-Labs/FcF-Inpainting/blob/main/colab/FcF_Inpainting.ipynb) [![Huggingface space](https://img.shields.io/badge/🤗-Huggingface%20Space-cyan.svg)](https://huggingface.co/spaces/shi-lab/FcF-Inpainting)  -->
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/) 
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This repo hosts the official implementary of:

[Xingqian Xu](https://ifp-uiuc.github.io/), Shant Navasardyan, Vahram Tadevosyan, Andranik Sargsyan, Yadong Mu and [Humphrey Shi](https://www.humphreyshi.com/home), **Image Completion with Heterogeneously Filtered Spectral Hints**, [ArXiv Link] coming soon.

## News

- [2022.11.07]: Our paper is accepted in WACV23. Will update code after CVPR deadline.
- [2022.11.06]: Repo initiated.

## Introduction

<p align="center">
  <img src="assets/teaser.png" width="99%">
</p>

Spectral Hint GAN (**SH-GAN**) is an high-performing inpainting network enpowered by CoModGAN and novel spectral processing techniques. SH-GAN reaches state-of-the-art on FFHQ and Places2 with freeform masks.

## Networks and Algorithms

The overall structure of our SH-GAN shows in the following figure:

<p align="center">
  <img src="assets/network.png" width="99%">
</p>

The sturcture of our Spectral Hint Unit shows in the following graph:

<p align="center">
  <img src="assets/shu.png" width="40%">
</p>

An explanation of Heterogeneous Filtering

<p align="center">
  <img src="assets/hfilter.png" width="80%">
</p>

An explanation of Gaussian Split Algorithm

<p align="center">
  <img src="assets/split.png" width="99%">
</p>

## Setup

coming soon

## Evaluation

coming soon

## Training

coming soon

## Citation

```BibTeX
```

## Acknowledgement

Part of the codes reorganizes/reimplements code from the following repositories: [Comodgan official Github](https://github.com/zsyzzsoft/co-mod-gan) and [Stylegan2-ADA official Github](https://github.com/NVlabs/stylegan2-ada-pytorch/).
