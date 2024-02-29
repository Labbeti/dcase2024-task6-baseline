# dcase2024-task6-baseline

<center>

<a href="https://www.python.org/">
    <img alt="Python" src="https://img.shields.io/badge/-Python 3.11-blue?style=for-the-badge&logo=python&logoColor=white">
</a>
<a href="https://pytorch.org/get-started/locally/">
    <img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 2.2-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white">
</a>
<a href="https://black.readthedocs.io/en/stable/">
    <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray">
</a>

DCASE2024 Challenge Task 6 baseline system (Automated Audio Captioning)

</center>

## Installation
First, you need to create an environment that contains **python>=3.11** and **pip**. You can use conda, mamba, micromamba or any other tool.

Here is an example with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
```bash
micromamba env create -n env_dcase24 python=3.11 pip -c defaults
micromamba activate env_dcase24
```

Then, you can clone this repository and install the requirements with pip:
```bash
git clone https://github.com/Labbeti/dcase2024-task6-baseline
pip install -e dcase2024-task6-baseline
```

## Usage

### Download external data, models and prepare
```bash
dcase24t6-prepare
```

### Train a model
```bash
dcase24t6-train model=baseline
```

### Test a model
```bash
dcase24t6-test model=baseline ckpt_path=last
```

## Code overview
This repository extensively use [PyTorch Lightning]() and [Hydra](). It is highly recommanded to learn about them if you want to understand this code.

Installation has three main steps:
- Download external models required for AAC metrics using [aac-metrics]()
- Download Clotho dataset using [aac-datasets]()
- Create HDF files containing Clotho subsets with preprocessed audio

Training follows the standard way to create a model with lightning:
- C


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
