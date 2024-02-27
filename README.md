# dcase2024-task6-baseline
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

## Project architecture
TODO

## Contact
Maintainer:
- [Étienne Labbé](https://labbeti.github.io/) "Labbeti": labbeti.pub@gmail.com
