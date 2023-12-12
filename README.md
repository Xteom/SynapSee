# SynapSee

SynapSee is an innovative project exploring the use of EEG signals for image categorization. This repository contains all the necessary code and data for replicating and extending our research.

## How to clone this repository

```bash
git clone http://github.com/NeuroLab-ITAM/SynapSee.git
cd SynapSee_data
git submodule init
git submodule update
```

## Setup environment

This project was developed in python 3.10.13.

```bash
conda create -n SynapSee python=3.10.13
conda activate SynapSee
pip install -r requirements.txt
```

## Folder structure

* `/images`: Folder with the images used in the app/experiments. You can find the link to the images in the images/README.md file.
* `/SynapSee_data`: Folder with the dataset or where the new datasets will be created.
* `/processed_data`: Folder with the processed data (cropped signal) from the raw data. (it was only used before the databuilder.ipynb was created). Some code in `SynapSee_eda.ipynb` still uses this folder but you can just go to `SynapSee_eda-II.ipynb` and use the json file.
* `/utils`: Folder with some utility functions.

## Code structure

The file are ordered as you should run (or look at) them to replicate the project. You should only run the files with the .ipynb extension.
The files are:

* `SynapSee_exp.ipynb`: Notebook to run the experiments.
* `SynapSee_exp.py`: Python file with the experiment's app.
* `databuilder.ipynb`: Notebook to build the dataset from the raw data. (output: `SynapSee_data.json`)
* `SynapSee_eda.ipynb`: Notebook to explore the dataset csv.
* `SynapSee_eda-II.ipynb`: Notebook to explore the dataset from json. (practically the same as `SynapSee_eda.ipynb`)
* `SynapSee_models1.ipynb`: Notebook where the first baseline model was created.
* `EEGDataset.py`: Python file with the EEGDataset class. This class turns the json dataset file that was created with the databuilder.ipynb into a pytorch dataset.
* `SynapSee_models2.ipynb`: Notebook where we explore multiple architectures and hyperparameters.
* `SynapSee_models3.ipynb`: Notebook where we will fine tune the best models from `SynapSee_models2.ipynb`.

## Run with cuda (this section has worked to be done)

Check cuda version

```bash
nvidia-smi.exe
```

Activate environment

```bash
conda activate SynapSee
```

Install cuda toolkit version

```bash
conda install cuda --channel nvidia/label/cuda-12.2.0
```

Check installation

```bash
import torch
print(f'PyTorch version: {torch.__version__}')
print('*'*10)
print(f'_CUDA version: ')
!nvcc --version
print('*'*10)
print(f'CUDNN version: {torch.backends.cudnn.version()}')
print(f'Available GPU devices: {torch.cuda.device_count()}')
print(f'Device Name: {torch.cuda.get_device_name()}')
```
