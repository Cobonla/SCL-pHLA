<h1 align="center">
    SCL-pHLA
    <br>
<h1>

<h4 align="center">Standalone program for the SCL-pHLA framework</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/Cobonla/SCL-pHLA?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/Cobonla/SCL-pHLA?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/Cobonla/SCL-pHLA?" alt="license"></a>
<a href="https://doi.org/10.5281/zenodo.16917568">
    <img src="https://zenodo.org/badge/doi/10.5281/zenodo.16917568.svg" alt="DOI">
</a>
</p>

<p align="center">
  <a href="#abstract">Abstract</a> •
  <a href="#introduction">Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#citation">Citation</a> •
</p>

# Abstract
Update soon!

# Introduction
This repository provides the standalone program for SCL-pHLA framework. The final models are available via Zenodo at 
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.16917568.svg)](https://doi.org/10.5281/zenodo.16917568)

# Installation
## Software requirements
* Ubuntu 20.04.6 LTS (This source code has been already tested on Ubuntu 20.04.6 LTS with NVIDIA T400 4GB)
* CUDA 12.2 (with GPU suport)
* cuDNN 8.9.7 (with GPU support)
* Python 3.9.18

### Creating conda environments
```shell
conda create -n SCL-pHLA python=3.9.18
```
```shell
conda activate SCL-pHLA
```

### Installing required specific packages
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

```shell
python -m pip install termcolor==2.3.0 --no-cache-dir
```

```shell
python -m pip install transformers==4.25.1 --no-cache-dir
```

```shell
python -m pip install numpy==1.26.4 pandas==1.5.3 scikit-learn==0.24.2 --no-cache-dir
```

### Getting started
```
git clone https://github.com/Cobonla/SCL-pHLA.git
```
```
cd SCL-pHLA
```

### Downloading final models
* For the Final_models.zip file, please download it via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.16917568.svg)](https://doi.org/10.5281/zenodo.16917568) and extract it into Final_models (https://github.com/Cobonla/SCL-pHLA/tree/main/Final_models) folder.

### Running prediction
```
python Pretrain.py
```

```
python Prediction.py
```
## Citation
If you use this code or part of it, please cite the following papers:
```
@article{x,
  title={SCL-pHLA: Integrating Feature Representation Learning and Pre-Trained Language Models for Predicting HLA-I-Peptide Binding with Experimental Validation},
  author={Phan, Le Thi and Shah, Masaud and Pham, Nhat Truong and Woo, Hyun Goo and Manavalan, Balachandran},
  journal={x},
  volume={x},
  pages={x},
  year={x},
  publisher={x}
}
```
