# dcase2024-task6-baseline

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

To download, extract and process data, you need to run:
```bash
dcase24t6-prepare
```
By default, the dataset is stored in `./data` directory. It will requires approximatively 33GB of disk space.

### Train the default model
```bash
dcase24t6-train
```

By default, the model and results are saved in directory `./logs/SAVE_NAME`. `SAVE_NAME` is the name of the script with the train starting date.

### Test the default model
```bash
dcase24t6-test ckpt_path=./logs/SAVE_NAME/checkpoints/MODEL.ckpt
```

## Code overview
This repository extensively use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training and [Hydra](https://hydra.cc/) for configuration.
It is highly recommanded to learn about them if you want to understand this code.

Installation has three main steps:
- Download external models (required for metrics and audio features)
- Download Clotho dataset using [aac-datasets](https://github.com/Labbeti/aac-datasets)
- Create HDF files containing each Clotho subset with preprocessed audio features

Training follows the standard way to create a model with lightning:
- Initialize callbacks, tokenizer, datamodule, model.
- Start fitting the model on the specified datamodule.
- Evaluate the model using [aac-metrics](https://github.com/Labbeti/aac-metrics)


## See also
- [Task description page](https://dcase.community/challenge2024/task-automated-audio-captioning)
- [DCASE2023 Audio Captioning baseline](https://github.com/felixgontier/dcase-2023-baseline)
- [DCASE2022 Audio Captioning baseline](https://github.com/felixgontier/dcase-2022-baseline)
- [DCASE2021 Audio Captioning baseline](https://github.com/audio-captioning/dcase-2021-baseline)
- [DCASE2020 Audio Captioning baseline](https://github.com/audio-captioning/dcase-2020-baseline)
- [aac-datasets](https://github.com/Labbeti/aac-datasets)
- [aac-metrics](https://github.com/Labbeti/aac-metrics)


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
