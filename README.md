# SynapSee

SynapSee is an innovative project exploring the use of EEG signals for image categorization. This repository contains all the necessary code and data for replicating and extending our research.

## How to clone this repository

```bash
git clone --recursive-submodules https://github.com/NeuroLab-ITAM/SynapSee.git
```

or

```bash
git clone http://github.com/NeuroLab-ITAM/SynapSee.git
cd SynapSee_data
git submodule init
git submodule update
```

## Code structure

* databuilder.ipynb: Notebook to build the dataset from the raw data.
* SynapSee_exp.ipynb: Notebook to run the experiments.
* SynapSee_exp.py: Python file to run the experiments.
* SynapSee_eda.ipynb: Notebook to explore the dataset csv.
* SynapSee_eda-II.ipynb: Notebook to explore the dataset from json.
* /images: Folder with the images used in the experiments.
* /SynapSee_data: Folder with the dataset.
* LSTM-CNN.ipynb: Notebook with the LSTM-CNN model
* SynapSee_models.ipynb: Notebook where we are currently working on new models.


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