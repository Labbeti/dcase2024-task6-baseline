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

The main model is composed of a pretrained convolutional encoder to extract features and a transformer decoder to generate caption.
For more information, please refer to the corresponding [DCASE task page](https://dcase.community/challenge2024/task-automated-audio-captioning).


## Installation
First, you need to create an environment that contains **python>=3.11** and **pip**. You can use venv, conda, micromamba or any python environment manager tool.

Here is an example with [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html):
```bash
micromamba env create -n env_dcase24 python=3.11 pip -c defaults
micromamba activate env_dcase24
```

Then, you can clone this repository and install it:
```bash
git clone https://github.com/Labbeti/dcase2024-task6-baseline
cd dcase2024-task6-baseline
pip install -e .
pre-commit install
```

You also need to install Java >= 1.8 and <= 1.13 on your machine to compute AAC metrics. If needed, you can override java executable path with the environment variable `AAC_METRICS_JAVA_PATH`.

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
The source code extensively use [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) for training and [Hydra](https://hydra.cc/) for configuration.
It is highly recommanded to learn about them if you want to understand this code.

Installation has three main steps:
- Download external models (like [ConvNeXt](https://github.com/topel/audioset-convnext-inf) to extract audio features)
- Download Clotho dataset using [aac-datasets](https://github.com/Labbeti/aac-datasets)
- Create HDF files containing each Clotho subset with preprocessed audio features

Training follows the standard way to create a model with lightning:
- Initialize callbacks, tokenizer, datamodule, model.
- Start fitting the model on the specified datamodule.
- Evaluate the model using [aac-metrics](https://github.com/Labbeti/aac-metrics)

## Model

The model outperforms previous baselines with a SPIDEr-FL score of 29.5% on the Clotho evaluation subset.
The architecture is described in [this paper](https://arxiv.org/pdf/2309.00454.pdf) and called **CNext-trans**. The encoder part (ConvNeXt) is described in more detail in [this paper](https://arxiv.org/pdf/2306.00830.pdf).

### Main hyperparameters

| Hyperparameter | Value | Option |
| --- | --- | --- |
| Number of epochs | 400 | `trainer.max_epochs` |
| Learning rate | 5e-4 | `model.lr` |
| Weight decay | 2 | `model.weight_decay` |
| Gradient clipping | 1 | `trainer.gradient_clip_val` |
| Beam size | 3 | `model.beam_size` |
| Model dimension size | 256 | `model.d_model` |
| Label smoothing | 0.2 | `model.label_smoothing` |
| Mixup alpha | 0.4 | `model.mixup_alpha` |

<!-- TODO: model size -->

## Tips
- **Modify the model**
The model class is located in `src/dcase24t6/models/trans_decoder.py`. It is recommanded to create another class and conf to keep different models architectures.
The loss is computed in the method called `training_step`. You can also modify the model architecture in the method called `setup`.

- **Extract different audio features**
For that, you can add a new pre-process function in `src/dcase24t6/pre_processes` and the related conf in `src/conf/pre_process`. Then, re-run `dcase24t6-prepare pre_process=YOUR_PROCESS download_clotho=false` to create new HDF files with your own features.
To train a new model on these features, you can specify the HDF files required in `dcase24t6-train datamodule.train_hdfs=clotho_dev_YOUR_PROCESS.hdf datamodule.val_hdfs=... datamodule.test_hdfs=... datamodule.predict_hdfs=...`. Depending on the features extracted, some parameters could be modified in the model to handle them.

- **Using as a package**
If you do not want ot use the entire codebase but only parts of it, you can install it as a package using:

```bash
pip install git+https://github.com/Labbeti/dcase2024-task6-baseline
```

Then you will be able to import any object from the code like for example `from dcase24t6.models.trans_decoder import TransDecoderModel`. There is also several important dependencies that you can install separately:

- `aac-metrics` to compute AAC metrics,
- `aac-datasets` to download and load AAC datasets,
- `torchoutil[extras]` to pack to HDF datasets.


## Additional information
- The code has been made for Ubuntu 20.04 and should work on more recent Ubuntu versions.
- The GPU used is NVIDIA GeForce RTX 2080 Ti. Training lasts for approximatively 2 hours in the default setting.
- In this code, the clotho subsets are naming following the Clotho convention, not DCASE convention. See more information [on this page](https://aac-datasets.readthedocs.io/en/stable/data_subsets.html#clotho).


## See also
- [DCASE2023 Audio Captioning baseline](https://github.com/felixgontier/dcase-2023-baseline)
- [DCASE2022 Audio Captioning baseline](https://github.com/felixgontier/dcase-2022-baseline)
- [DCASE2021 Audio Captioning baseline](https://github.com/audio-captioning/dcase-2021-baseline)
- [DCASE2020 Audio Captioning baseline](https://github.com/audio-captioning/dcase-2020-baseline)
- [aac-datasets](https://github.com/Labbeti/aac-datasets)
- [aac-metrics](https://github.com/Labbeti/aac-metrics)


## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
